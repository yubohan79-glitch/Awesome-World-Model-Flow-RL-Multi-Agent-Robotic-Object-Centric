from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]




def _arena_source() -> str:
    """Concatenate all split arena-sim submodules for source-contract checks."""
    pkg = ROOT / "isaaclab_sim" / "robocup_visionrl_arena_sim"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(pkg.glob("*.py")))

def test_public_target_layout_matches_rule_contract():
    layout = yaml.safe_load((ROOT / "config" / "target_layout.yaml").read_text(encoding="utf-8"))
    normals = layout["targets"]["normal"]["positions"]
    base = layout["targets"]["base"]

    assert layout["targets"]["normal"]["tag_id"] == 1
    assert len(normals) == 8
    assert base["yellow"]["tag_id"] == 2
    assert base["blue"]["tag_id"] == 3
    assert base["yellow"]["pitch_deg"] == 0
    assert base["blue"]["pitch_deg"] == 0
    assert base["yellow"]["face_yaw_deg"] == 135
    assert base["blue"]["face_yaw_deg"] == -45
    assert base["yellow"]["plane_angle_to_walls_deg"] == 45
    assert base["blue"]["plane_angle_to_walls_deg"] == 45
    assert abs(base["yellow"]["yaw"] - 2.3562) < 1e-4
    assert abs(base["blue"]["yaw"] + 0.7854) < 1e-4
    assert {target["owner"] for target in normals} == {"yellow", "blue"}
    by_name = {target["name"]: target for target in normals}
    assert by_name["T01_NorthMiddle"]["xy"] == [0.18, 1.26]
    assert by_name["T08_SouthMiddle"]["xy"] == [-0.18, -1.26]
    assert abs(by_name["T01_NorthMiddle"]["yaw"] + 0.7854) < 1e-4
    assert abs(by_name["T08_SouthMiddle"]["yaw"] - 2.3562) < 1e-4
    assert by_name["T03_WestAboveGate"]["xy"] == [-1.26, 0.24]
    assert by_name["T06_EastBelowGate"]["xy"] == [1.26, -0.24]


def test_isaaclab_base_targets_are_vertical_not_wall_leaning():
    source = _arena_source()

    assert "pitch=math.radians(-45.0)" not in source
    assert "BLUE_BASE_TARGET_XY = (-1.36, 1.36)" in source
    assert "YELLOW_BASE_TARGET_XY = (1.36, -1.36)" in source
    assert "NORTH_MIDDLE_TARGET_X = 0.18" in source
    assert "SOUTH_MIDDLE_TARGET_X = -0.18" in source
    assert "SIDE_GATE_TARGET_Y = 0.24" in source
    assert "TARGET_WALL_INSET = 0.240" in source
    assert "BLUE_BASE_TARGET_YAW = -math.pi / 4.0" in source
    assert "YELLOW_BASE_TARGET_YAW = 3.0 * math.pi / 4.0" in source
    assert "tag_local_z = TAG_CENTER_Z - board_center[2]" in source
    assert "support_height = 0.115 if base_target else 0.120" in source
    assert "support_offset_x = 0.034 if base_target else -0.034" in source
    assert "foot_offset_x = 0.045 if base_target else -0.045" in source


def test_isaaclab_base_armor_uses_rule_edge_l_shape():
    source = _arena_source()

    assert "YELLOW_ATTACK_BLUE_BASE_XY = (-0.72, 1.32)" in source
    assert "BLUE_ATTACK_YELLOW_BASE_XY = (0.72, -1.32)" in source
    assert '("armor_1", (-1.025, 1.375, z), (armor_thickness, armor_length, armor_height))' in source
    assert '("armor_2", (-1.375, 1.025, z), (armor_length, armor_thickness, armor_height))' in source
    assert '("armor_3", (-1.025, 1.125, z), (armor_thickness, armor_length, armor_height))' in source
    assert '("armor_4", (-1.125, 1.025, z), (armor_length, armor_thickness, armor_height))' in source
    assert "right edge plates 1/3 and lower edge plates 2/4" in source
    assert '("BlueBaseTarget", (-1.272, 0.807))' not in source
    assert '("YellowBaseTarget", (1.272, -0.807))' not in source


def test_isaaclab_pushable_obstacles_use_rule_diagram_reference_positions():
    source = _arena_source()

    assert '"box_ne": (0.80, 0.80)' in source
    assert '"box_sw": (-0.80, -0.80)' in source
    assert '("RandomObstacleNorthEast", (*PUSHABLE_OBSTACLE_STARTS["box_ne"], OBSTACLE_SIZE * 0.5))' in source
    assert '("RandomObstacleSouthWest", (*PUSHABLE_OBSTACLE_STARTS["box_sw"], OBSTACLE_SIZE * 0.5))' in source
    assert '("RandomObstacleNorthEast", (0.40, 0.48, OBSTACLE_SIZE * 0.5))' not in source
    assert '("RandomObstacleSouthWest", (-0.40, -0.48, OBSTACLE_SIZE * 0.5))' not in source


def test_isaaclab_pushable_obstacles_are_dynamic_physical_boxes():
    source = _arena_source()
    body = source[source.index("def spawn_pushable_obstacle") : source.index("def segment_intersects_aabb")]

    assert "rigid_body=True" in body
    assert "kinematic=False" in body
    assert "disable_gravity=False" in body
    assert "mass=PUSHABLE_OBSTACLE_MASS_KG" in body
    assert "physics_material=rigid_physics_material" in body
    assert "PUSHABLE_OBSTACLE_STATIC_FRICTION" in body
    assert "PUSHABLE_OBSTACLE_DYNAMIC_FRICTION" in body
    assert "contact_offset=0.010" in body
    assert "sync_pushable_obstacles_from_stage()" in source


def test_isaaclab_replay_keeps_pushed_box_pose_when_trace_is_static():
    source = _arena_source()

    assert '"last_box_trace_xy"' in source
    assert "trace_changed =" in source
    assert "if not trace_changed:" in source
    assert "continue" in source[source.index("if not trace_changed:") : source.index("def point_blocked")]


def test_isaaclab_replay_uses_visual_hull_for_pushable_boxes():
    source = _arena_source()

    assert "ROBOT_PUSHABLE_CLEARANCE_RADIUS = ROBOT_COLLISION_RADIUS + 0.030" in source
    assert "ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS = ROBOT_PUSHABLE_CLEARANCE_RADIUS + 0.065" in source
    assert "ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS = (ROBOT_LENGTH * 0.5 + 0.110, ROBOT_WIDTH * 0.5 + WHEEL_WIDTH + 0.062)" in source
    assert "ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS" in source[source.index("def dynamic_pushable_costmap") : source.index("def aabb_clearance")]
    assert "visual hull separated after trace correction" in source
    assert "for obstacle_path, center, half_size in dynamic_pushable_costmap():" in source


def test_isaaclab_recorder_supports_true_top_view():
    source = _arena_source()

    assert 'choices=["overview", "top", "yellow_pov", "blue_pov"]' in source
    assert 'self.view not in ("overview", "top")' in source
    assert 'return "Top View"' in source
    assert "RECORDING_POV_CAMERA_POSE" in source
    assert "horizontal_aperture=7.2" in source


def test_elimination_target_files_are_team_separated():
    yellow_config = yaml.safe_load((
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_navigation"
        / "config"
        / "targets.elimination.yellow.yaml"
    ).read_text(encoding="utf-8"))
    blue_config = yaml.safe_load((
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_navigation"
        / "config"
        / "targets.elimination.blue.yaml"
    ).read_text(encoding="utf-8"))

    yellow_owners = yellow_config["competition_behavior"]["ros__parameters"]["target_owner"]
    blue_owners = blue_config["competition_behavior"]["ros__parameters"]["target_owner"]

    assert set(yellow_owners) == {"blue"}
    assert set(blue_owners) == {"yellow"}


def test_ros2_elimination_routes_match_current_isaaclab_standoffs():
    nav_config = ROOT / "crc_robocup_vision_ws" / "src" / "rcvrl_navigation" / "config"
    yellow = yaml.safe_load((nav_config / "targets.elimination.yellow.yaml").read_text(encoding="utf-8"))
    blue = yaml.safe_load((nav_config / "targets.elimination.blue.yaml").read_text(encoding="utf-8"))
    default = yaml.safe_load((nav_config / "targets.elimination.yaml").read_text(encoding="utf-8"))

    yellow_params = yellow["competition_behavior"]["ros__parameters"]
    blue_params = blue["competition_behavior"]["ros__parameters"]
    default_params = default["competition_behavior"]["ros__parameters"]

    expected_yellow = {
        "target_name": ["T01_NorthMiddle", "T03_WestAboveGate", "T05_EastAboveGate", "T02_NorthEast", "BlueBaseTarget"],
        "target_owner": ["blue", "blue", "blue", "blue", "blue"],
        "target_x": [1.92, 0.30, 2.70, 2.68, 0.23],
        "target_y": [2.52, 1.84, 1.84, 2.68, 2.31],
        "target_yaw": [2.36, -2.36, -0.79, 0.79, 1.73],
        "target_tag_id": [1, 1, 1, 1, 3],
    }
    expected_blue = {
        "target_name": ["T08_SouthMiddle", "T04_WestBelowGate", "T06_EastBelowGate", "T07_SouthWest", "YellowBaseTarget"],
        "target_owner": ["yellow", "yellow", "yellow", "yellow", "yellow"],
        "target_x": [1.08, 0.30, 2.70, 0.32, 2.77],
        "target_y": [0.48, 1.16, 1.16, 0.32, 0.69],
        "target_yaw": [-0.79, 2.36, 0.79, -2.36, -1.41],
        "target_tag_id": [1, 1, 1, 1, 2],
    }

    for key, expected in expected_yellow.items():
        assert yellow_params[key] == expected
        assert default_params[key] == expected
    for key, expected in expected_blue.items():
        assert blue_params[key] == expected


def test_ros2_qualifier_route_is_not_stale_and_contains_all_targets():
    params = yaml.safe_load((
        ROOT
        / "crc_robocup_vision_ws"
        / "src"
        / "rcvrl_navigation"
        / "config"
        / "targets.qualifier.yaml"
    ).read_text(encoding="utf-8"))["competition_behavior"]["ros__parameters"]

    assert params["target_name"] == [
        "T01_NorthMiddle",
        "T02_NorthEast",
        "T05_EastAboveGate",
        "T06_EastBelowGate",
        "T07_SouthWest",
        "T04_WestBelowGate",
        "T03_WestAboveGate",
        "T08_SouthMiddle",
        "BlueBaseTarget",
    ]
    assert params["target_owner"] == ["unknown"] * 8 + ["blue"]
    assert params["target_x"] == [1.92, 2.68, 2.70, 2.70, 0.32, 0.30, 0.30, 1.08, 0.23]
    assert params["target_y"] == [2.52, 2.68, 1.84, 1.16, 0.32, 1.16, 1.84, 0.48, 2.31]
    assert params["target_yaw"] == [2.36, 0.79, -0.79, 0.79, -2.36, 2.36, -2.36, -0.79, 1.73]
    assert params["target_tag_id"] == [1, 1, 1, 1, 1, 1, 1, 1, 3]


def test_behavior_cpp_fallback_route_is_not_stale_qualifier_layout():
    source = (ROOT / "crc_robocup_vision_ws" / "src" / "rcvrl_behavior" / "src" / "competition_behavior.cpp").read_text(encoding="utf-8")

    assert '"target_x", {1.92, 0.30, 2.70, 2.68, 0.23}' in source
    assert '"target_y", {2.52, 1.84, 1.84, 2.68, 2.31}' in source
    assert '"target_yaw", {2.36, -2.36, -0.79, 0.79, 1.73}' in source
    assert '"target_tag_id", {1, 1, 1, 1, 3}' in source
    assert "T01_NorthMiddle" in source
    assert "BlueBaseTarget" in source


def test_strict_replay_renderer_draws_dynamic_boxes_and_armor():
    source = (ROOT / "isaaclab_sim" / "rl" / "render_strict_replay_video.py").read_text(encoding="utf-8")

    assert "active_armor_for_row" in source
    assert "pushable_boxes_for_row" in source
    assert "STATIC_BLOCKERS" in source
    assert "BASE_ARMOR_BLOCKERS" in source
    assert "for center, span in active_armor_for_row(rows[current_step])" in source
    assert "for center, span in pushable_boxes_for_row(rows[current_step])" in source
