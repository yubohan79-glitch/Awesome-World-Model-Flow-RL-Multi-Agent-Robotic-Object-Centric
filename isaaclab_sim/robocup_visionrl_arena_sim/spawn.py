from __future__ import annotations



from ._bootstrap import (
    BASE_ARMOR,
    COLLISION_PRIMS,
    LASER_BLOCKERS,
    NAV_BLOCKERS,
    PUSHABLE_OBSTACLES,
    PUSHABLE_OBSTACLE_DYNAMIC_FRICTION,
    PUSHABLE_OBSTACLE_MASS_KG,
    PUSHABLE_OBSTACLE_STATIC_FRICTION,
    RAYCAST_BOXES,
    ROUTE_CLEARANCE,
    TAG_CENTER_Z,
    TAG_SIZE,
    TARGET_REGISTRY,
    sim_utils
)
from .transforms import (
    create_xform,
    local_to_world,
    quat_from_euler,
    set_visibility
)

def material(
    color: tuple[float, float, float],
    opacity: float = 1.0,
    roughness: float = 0.55,
    metallic: float = 0.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=emissive,
        roughness=roughness,
        metallic=metallic,
        opacity=opacity,
    )


def rigid_physics_material(
    static_friction: float,
    dynamic_friction: float,
    restitution: float = 0.0,
) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def spawn_box(
    path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rigid_body: bool = False,
    kinematic: bool = False,
    mass: float | None = None,
    disable_gravity: bool = False,
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
):
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=material(color, opacity=opacity, emissive=emissive),
        physics_material=physics_material,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        )
        if collision or rigid_body
        else None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=kinematic,
            disable_gravity=disable_gravity,
            linear_damping=0.18 if not kinematic else None,
            angular_damping=0.75 if not kinematic else None,
            max_depenetration_velocity=1.2,
            solver_position_iteration_count=8 if not kinematic else None,
            solver_velocity_iteration_count=2 if not kinematic else None,
        )
        if rigid_body
        else None,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass) if rigid_body and mass is not None else None,
        semantic_tags=[("class", semantic)] if semantic else None,
    )
    cfg.func(path, cfg, translation=pos, orientation=orientation)
    if raycast:
        COLLISION_PRIMS.append(path)
        RAYCAST_BOXES.append((size, pos, orientation))


def spawn_cylinder(
    path: str,
    radius: float,
    height: float,
    axis: str,
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        axis=axis,
        visual_material=material(color, opacity=opacity, emissive=emissive),
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        semantic_tags=[("class", semantic)] if semantic else None,
    )
    cfg.func(path, cfg, translation=pos, orientation=orientation)
    if raycast:
        COLLISION_PRIMS.append(path)
        if axis.upper() == "X":
            box_size = (height, radius * 2.0, radius * 2.0)
        elif axis.upper() == "Y":
            box_size = (radius * 2.0, height, radius * 2.0)
        else:
            box_size = (radius * 2.0, radius * 2.0, height)
        RAYCAST_BOXES.append((box_size, pos, orientation))


def spawn_marker_cell(
    path: str,
    center: tuple[float, float, float],
    size_y: float,
    size_z: float,
    local_y: float,
    local_z: float,
    roll: float,
    pitch: float,
    yaw: float,
    color: tuple[float, float, float] = (0.01, 0.01, 0.01),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, (0.003, local_y, local_z), roll, pitch, yaw)
    spawn_box(path, (0.003, size_y, size_z), pos, color, orientation=orient, semantic="apriltag_visual")


def spawn_local_box(
    path: str,
    center: tuple[float, float, float],
    local_offset: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    *,
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, local_offset, roll, pitch, yaw)
    spawn_box(
        path,
        size,
        pos,
        color,
        orientation=orient,
        collision=collision,
        raycast=raycast,
        semantic=semantic,
        opacity=opacity,
        emissive=emissive,
    )


def spawn_local_cylinder(
    path: str,
    center: tuple[float, float, float],
    local_offset: tuple[float, float, float],
    radius: float,
    height: float,
    axis: str,
    color: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    *,
    collision: bool = False,
    raycast: bool = False,
    semantic: str | None = None,
    opacity: float = 1.0,
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    orient = quat_from_euler(roll, pitch, yaw)
    pos = local_to_world(center, local_offset, roll, pitch, yaw)
    spawn_cylinder(
        path,
        radius,
        height,
        axis,
        pos,
        color,
        orientation=orient,
        collision=collision,
        raycast=raycast,
        semantic=semantic,
        opacity=opacity,
        emissive=emissive,
    )


def spawn_apriltag(
    path: str,
    center: tuple[float, float, float],
    tag_id: int,
    roll: float,
    pitch: float,
    yaw: float,
):
    """Build a physical tag-like target from geometry.

    The high-contrast layout is intentionally made of primitive geometry so the
    USD stays portable. The metadata and surrounding docs record that the real
    detector uses the AprilTag Tag36h11 family.
    """
    create_xform(path)
    orient = quat_from_euler(roll, pitch, yaw)

    spawn_box(
        f"{path}/black_carrier",
        (0.002, TAG_SIZE * 1.28, TAG_SIZE * 1.28),
        local_to_world(center, (-0.001, 0.0, 0.0), roll, pitch, yaw),
        (0.01, 0.012, 0.012),
        orientation=orient,
        semantic=f"tag36h11_id_{tag_id}_carrier",
    )
    spawn_box(
        f"{path}/white_laminate",
        (0.003, TAG_SIZE * 1.14, TAG_SIZE * 1.14),
        center,
        (0.97, 0.97, 0.93),
        orientation=orient,
        semantic=f"tag36h11_id_{tag_id}",
    )

    border = TAG_SIZE * 0.13
    half = TAG_SIZE * 0.5
    spawn_marker_cell(f"{path}/border_left", center, border, TAG_SIZE, -half + border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_right", center, border, TAG_SIZE, half - border * 0.5, 0.0, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_top", center, TAG_SIZE, border, 0.0, half - border * 0.5, roll, pitch, yaw)
    spawn_marker_cell(f"{path}/border_bottom", center, TAG_SIZE, border, 0.0, -half + border * 0.5, roll, pitch, yaw)

    # Compact 6x6 visual code. It is not used for detection in this script; it
    # makes IDs 1, 2, and 3 visibly distinct while still reading like AprilTag.
    patterns_by_id = {
        1: {
            (0, 0),
            (0, 5),
            (1, 1),
            (1, 4),
            (2, 2),
            (2, 5),
            (3, 0),
            (3, 3),
            (4, 1),
            (4, 4),
            (5, 2),
            (5, 5),
        },
        2: {
            (0, 1),
            (0, 4),
            (1, 0),
            (1, 2),
            (2, 3),
            (2, 5),
            (3, 1),
            (3, 4),
            (4, 0),
            (4, 3),
            (5, 2),
            (5, 4),
        },
        3: {
            (0, 0),
            (0, 3),
            (0, 5),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 2),
            (2, 5),
            (3, 1),
            (3, 3),
            (4, 0),
            (4, 2),
            (4, 5),
            (5, 1),
            (5, 4),
        },
    }
    cell = TAG_SIZE * 0.075
    pitch_between = TAG_SIZE * 0.112
    origin = -2.5 * pitch_between
    for iy, iz in patterns_by_id.get(tag_id, patterns_by_id[1]):
        local_y = origin + iy * pitch_between
        local_z = origin + iz * pitch_between
        spawn_marker_cell(
            f"{path}/id_{tag_id}_cell_{iy}_{iz}",
            center,
            cell,
            cell,
            local_y,
            local_z,
            roll,
            pitch,
            yaw,
        )


def spawn_target_id_badge(
    path: str,
    board_center: tuple[float, float, float],
    tag_id: int,
    board_size: tuple[float, float, float],
    roll: float,
    pitch: float,
    yaw: float,
    accent_color: tuple[float, float, float],
):
    front_x = board_size[0] * 0.5 + 0.008
    badge_y = -board_size[1] * 0.5 + 0.045
    badge_z = board_size[2] * 0.5 - 0.030
    spawn_local_box(
        f"{path}/id_badge_backplate",
        board_center,
        (front_x, badge_y, badge_z),
        (0.006, 0.070, 0.028),
        (0.04, 0.045, 0.045),
        roll,
        pitch,
        yaw,
        semantic="target_id_badge",
    )
    spawn_local_box(
        f"{path}/id_badge_team_strip",
        board_center,
        (front_x + 0.002, badge_y, badge_z + 0.010),
        (0.007, 0.062, 0.006),
        accent_color,
        roll,
        pitch,
        yaw,
        semantic="target_id_team_strip",
    )
    for index in range(tag_id):
        dot_y = badge_y - 0.018 + index * 0.018
        spawn_local_box(
            f"{path}/id_badge_dot_{index + 1}",
            board_center,
            (front_x + 0.004, dot_y, badge_z - 0.004),
            (0.008, 0.010, 0.012),
            (0.95, 0.95, 0.86),
            roll,
            pitch,
            yaw,
            semantic=f"target_id_{tag_id}_dot",
        )


def spawn_target(
    path: str,
    xy: tuple[float, float],
    yaw: float,
    *,
    tag_id: int = 1,
    pitch: float = 0.0,
    frame_color: tuple[float, float, float] = (0.20, 0.22, 0.24),
    base_target: bool = False,
):
    create_xform(path)
    roll = 0.0
    orient = quat_from_euler(roll, pitch, yaw)
    accent_color = frame_color
    face_color = (0.86, 0.87, 0.80)
    dark_frame = (0.035, 0.038, 0.040)
    warning_color = (0.98, 0.70, 0.12) if tag_id == 1 else frame_color
    board_center = (xy[0], xy[1], 0.116 if base_target else 0.116)
    board_size = (0.012, 0.095, 0.115) if base_target else (0.012, 0.180, 0.190)
    front_x = board_size[0] * 0.5 + 0.006
    edge = 0.010 if base_target else 0.010
    # The rules put the bottom of the 5 cm AprilTag at 6.5-7.5 cm above
    # the floor. Keep the visual tag anchored to that physical height instead
    # of drifting upward with the decorative board.
    tag_local_z = TAG_CENTER_Z - board_center[2]

    spawn_box(
        f"{path}/target_board",
        board_size,
        board_center,
        face_color,
        orientation=orient,
        collision=True,
        raycast=True,
        semantic=f"target_board_id_{tag_id}",
    )

    # Raised structural frame: it makes the target read as hardware instead of
    # a flat texture and gives the laser hit board a clear silhouette.
    spawn_local_box(
        f"{path}/frame_left",
        board_center,
        (front_x, -board_size[1] * 0.5 + edge * 0.5, 0.0),
        (0.009, edge, board_size[2] + edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_right",
        board_center,
        (front_x, board_size[1] * 0.5 - edge * 0.5, 0.0),
        (0.009, edge, board_size[2] + edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_top",
        board_center,
        (front_x, 0.0, board_size[2] * 0.5 - edge * 0.5),
        (0.009, board_size[1] + edge, edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )
    spawn_local_box(
        f"{path}/frame_bottom",
        board_center,
        (front_x, 0.0, -board_size[2] * 0.5 + edge * 0.5),
        (0.009, board_size[1] + edge, edge),
        dark_frame,
        roll,
        pitch,
        yaw,
        semantic="target_frame",
    )

    spawn_local_box(
        f"{path}/lower_status_strip",
        board_center,
        (front_x + 0.003, 0.0, -board_size[2] * 0.5 + edge + 0.010),
        (0.008, board_size[1] - edge * 2.4, 0.010),
        warning_color,
        roll,
        pitch,
        yaw,
        semantic="target_status_strip",
        emissive=(warning_color[0] * 0.05, warning_color[1] * 0.05, warning_color[2] * 0.05),
    )

    tag_center = local_to_world(board_center, (front_x + 0.004, 0.0, tag_local_z), roll, pitch, yaw)
    spawn_apriltag(f"{path}/tag36h11_{tag_id}", tag_center, tag_id, roll, pitch, yaw)

    reticle_color = (0.92, 0.08, 0.08) if not base_target else (0.98, 0.18, 0.18)
    for index, (local_y, local_z, size_y, size_z) in enumerate(
        (
            (-board_size[1] * 0.26, tag_local_z, 0.022, 0.004),
            (board_size[1] * 0.26, tag_local_z, 0.022, 0.004),
            (0.0, tag_local_z - board_size[2] * 0.26, 0.004, 0.022),
            (0.0, tag_local_z + board_size[2] * 0.26, 0.004, 0.022),
        )
    ):
        spawn_local_box(
            f"{path}/laser_reticle_{index + 1}",
            board_center,
            (front_x + 0.006, local_y, local_z),
            (0.005, size_y, size_z),
            reticle_color,
            roll,
            pitch,
            yaw,
            semantic="laser_hit_reticle",
            emissive=(0.10, 0.0, 0.0),
        )

    spawn_target_id_badge(path, board_center, tag_id, board_size, roll, pitch, yaw, accent_color)

    lens_y = board_size[1] * 0.5 - (0.030 if base_target else 0.035)
    lens_z = board_size[2] * 0.5 - (0.028 if base_target else 0.032)
    spawn_local_box(
        f"{path}/hit_indicator_lens",
        board_center,
        (front_x + 0.007, lens_y, lens_z),
        (0.009, 0.020, 0.020),
        (0.86, 0.02, 0.03),
        roll,
        pitch,
        yaw,
        semantic="laser_hit_indicator",
        emissive=(0.25, 0.0, 0.0),
    )

    support_height = 0.115 if base_target else 0.120
    # Base targets sit behind armor in a tight corner. Their low stand must
    # remain on the arena-facing side so it does not visually or physically
    # clip the grounded armor plates.
    support_offset_x = 0.034 if base_target else -0.034
    foot_offset_x = 0.045 if base_target else -0.045
    support_center = local_to_world(board_center, (support_offset_x, 0.0, -board_size[2] * 0.5 + support_height * 0.5), roll, pitch, yaw)
    spawn_box(
        f"{path}/rear_support_post",
        (0.018, 0.018, support_height),
        support_center,
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_rear_support",
    )
    spawn_local_cylinder(
        f"{path}/bottom_hinge",
        board_center,
        (support_offset_x, 0.0, -board_size[2] * 0.5 - 0.004),
        0.010,
        board_size[1] * 0.88,
        "Y",
        (0.08, 0.085, 0.085),
        roll,
        pitch,
        yaw,
        semantic="target_hinge",
    )
    foot_center = local_to_world(board_center, (foot_offset_x, 0.0, -board_center[2] + 0.014), roll, pitch, yaw)
    foot_size = (0.075, 0.115, 0.018) if base_target else (0.110, 0.205, 0.016)
    spawn_box(
        f"{path}/weighted_base_plate",
        foot_size,
        (foot_center[0], foot_center[1], foot_size[2] * 0.5),
        frame_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        collision=True,
        raycast=True,
        semantic="target_weighted_base",
    )
    for index, local_y in enumerate((-foot_size[1] * 0.36, foot_size[1] * 0.36)):
        spawn_local_cylinder(
            f"{path}/base_anchor_bolt_{index + 1}",
            (foot_center[0], foot_center[1], foot_size[2] * 0.5),
            (foot_size[0] * 0.24, local_y, foot_size[2] * 0.5 + 0.002),
            0.010,
            0.006,
            "Z",
            (0.05, 0.052, 0.052),
            0.0,
            0.0,
            yaw,
            semantic="target_base_bolt",
        )

    if base_target:
        spawn_local_box(
            f"{path}/base_target_backbone",
            board_center,
            (-0.030, 0.0, 0.0),
            (0.030, board_size[1] * 0.72, 0.020),
            dark_frame,
            roll,
            pitch,
            yaw,
            semantic="base_target_backbone",
        )
        spawn_local_box(
            f"{path}/base_target_warning_window",
            board_center,
            (front_x + 0.009, 0.0, board_size[2] * 0.5 - 0.064),
            (0.006, board_size[1] * 0.42, 0.014),
            (0.96, 0.08, 0.10),
            roll,
            pitch,
            yaw,
            semantic="base_target_critical_window",
            emissive=(0.18, 0.0, 0.0),
        )

    fallen_path = f"/World/Targets/Fallen/{path.rsplit('/', 1)[-1]}_fallen"
    fall_anim_path = f"/World/Targets/Falling/{path.rsplit('/', 1)[-1]}_falling"
    create_xform(fall_anim_path, translation=(xy[0], xy[1], 0.0), orientation=quat_from_euler(0.0, 0.0, yaw))
    spawn_box(
        f"{fall_anim_path}/target_board",
        board_size,
        (0.0, 0.0, board_center[2]),
        face_color,
        semantic=f"falling_target_board_id_{tag_id}",
    )
    spawn_box(
        f"{fall_anim_path}/target_frame_top",
        (0.010, board_size[1] + edge, edge),
        (front_x, 0.0, board_center[2] + board_size[2] * 0.5 - edge * 0.5),
        dark_frame,
        semantic="falling_target_frame",
    )
    spawn_box(
        f"{fall_anim_path}/target_frame_bottom",
        (0.010, board_size[1] + edge, edge),
        (front_x, 0.0, board_center[2] - board_size[2] * 0.5 + edge * 0.5),
        dark_frame,
        semantic="falling_target_frame",
    )
    spawn_apriltag(
        f"{fall_anim_path}/tag36h11_{tag_id}",
        (front_x + 0.006, 0.0, board_center[2] + tag_local_z),
        tag_id,
        0.0,
        0.0,
        0.0,
    )
    spawn_box(
        f"{fall_anim_path}/hit_indicator_lens",
        (0.008, 0.020, 0.020),
        (front_x + 0.012, lens_y, board_center[2] + lens_z),
        (0.86, 0.02, 0.03),
        emissive=(0.25, 0.0, 0.0),
        semantic="falling_hit_indicator",
    )
    spawn_box(
        f"{fall_anim_path}/weighted_base",
        foot_size,
        (-0.045, 0.0, foot_size[2] * 0.5),
        frame_color,
        semantic="falling_target_base",
    )
    spawn_cylinder(
        f"{fall_anim_path}/bottom_hinge",
        radius=0.010,
        height=board_size[1] * 0.88,
        axis="Y",
        pos=(-0.034, 0.0, board_center[2] - board_size[2] * 0.5 - 0.004),
        color=(0.08, 0.085, 0.085),
        semantic=f"falling_tag36h11_id_{tag_id}",
    )
    set_visibility(fall_anim_path, False)

    create_xform(fallen_path)
    spawn_box(
        f"{fallen_path}/board",
        (board_size[1], board_size[2], 0.014),
        (xy[0], xy[1], 0.022),
        face_color,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_target_id_{tag_id}",
    )
    spawn_box(
        f"{fallen_path}/tag_patch",
        (TAG_SIZE * 1.18, TAG_SIZE * 1.18, 0.005),
        (xy[0], xy[1], 0.034),
        (0.94, 0.94, 0.88),
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic=f"fallen_tag36h11_id_{tag_id}",
    )
    spawn_box(
        f"{fallen_path}/dark_frame",
        (board_size[1] + 0.018, board_size[2] + 0.018, 0.006),
        (xy[0], xy[1], 0.018),
        dark_frame,
        orientation=quat_from_euler(0.0, 0.0, yaw),
        semantic="fallen_target_frame_shadow",
    )
    set_visibility(fallen_path, False)

    if base_target:
        kind = "base_yellow" if tag_id == 2 else "base_blue"
    else:
        kind = "normal"
    if kind == "base_yellow":
        owner = "yellow"
    elif kind == "base_blue":
        owner = "blue"
    else:
        owner = "blue" if xy[1] >= 0.0 else "yellow"
    TARGET_REGISTRY[path] = {
        "path": path,
        "fallen_path": fallen_path,
        "fall_anim_path": fall_anim_path,
        "xy": xy,
        "yaw": yaw,
        "tag_id": tag_id,
        "kind": kind,
        "owner": owner,
        "knocked": False,
    }


def register_nav_blocker(path: str, pos: tuple[float, float, float], size: tuple[float, float, float]):
    margin = ROUTE_CLEARANCE + (0.045 if "BaseArmor" in path else 0.0)
    NAV_BLOCKERS.append(
        (
            path,
            (pos[0], pos[1]),
            (size[0] * 0.5 + margin, size[1] * 0.5 + margin),
        )
    )
    LASER_BLOCKERS.append((path, (pos[0], pos[1]), (size[0] * 0.5, size[1] * 0.5)))


def register_laser_blocker(path: str, pos: tuple[float, float, float], size: tuple[float, float, float]):
    LASER_BLOCKERS.append((path, (pos[0], pos[1]), (size[0] * 0.5, size[1] * 0.5)))


def unregister_blocker(path: str):
    NAV_BLOCKERS[:] = [item for item in NAV_BLOCKERS if item[0] != path]
    LASER_BLOCKERS[:] = [item for item in LASER_BLOCKERS if item[0] != path]


def spawn_nav_blocker(
    path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    semantic: str,
):
    spawn_box(path, size, pos, color, collision=True, raycast=True, semantic=semantic)
    register_nav_blocker(path, pos, size)


def register_pushable_obstacle(path: str, pos: tuple[float, float, float], size: tuple[float, float, float]):
    PUSHABLE_OBSTACLES[path] = {
        "xy": [float(pos[0]), float(pos[1])],
        "z": float(pos[2]),
        "size": size,
        "half": (size[0] * 0.5, size[1] * 0.5),
        "start_xy": (float(pos[0]), float(pos[1])),
        "last_push_t": -99.0,
        "rigid_body": True,
        "kinematic": False,
        "mass_kg": PUSHABLE_OBSTACLE_MASS_KG,
    }


def spawn_pushable_obstacle(
    path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    semantic: str,
):
    spawn_box(
        path,
        size,
        pos,
        color,
        collision=True,
        raycast=True,
        semantic=semantic,
        rigid_body=True,
        kinematic=False,
        mass=PUSHABLE_OBSTACLE_MASS_KG,
        disable_gravity=False,
        physics_material=rigid_physics_material(
            PUSHABLE_OBSTACLE_STATIC_FRICTION,
            PUSHABLE_OBSTACLE_DYNAMIC_FRICTION,
        ),
        contact_offset=0.010,
        rest_offset=0.001,
    )
    register_pushable_obstacle(path, pos, size)


def segment_intersects_aabb(
    p0: tuple[float, float],
    p1: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
) -> bool:
    min_x = center[0] - half_size[0]
    max_x = center[0] + half_size[0]
    min_y = center[1] - half_size[1]
    max_y = center[1] + half_size[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t_min = 0.0
    t_max = 1.0

    for start, delta, lower, upper in ((p0[0], dx, min_x, max_x), (p0[1], dy, min_y, max_y)):
        if abs(delta) < 1e-9:
            if start < lower or start > upper:
                return False
            continue
        inv_delta = 1.0 / delta
        t1 = (lower - start) * inv_delta
        t2 = (upper - start) * inv_delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def validate_route(name: str, route: list[tuple[float, float]], *, strict: bool = False) -> bool:
    valid = True
    for index, (p0, p1) in enumerate(zip(route, route[1:])):
        for blocker_path, center, half_size in NAV_BLOCKERS:
            if segment_intersects_aabb(p0, p1, center, half_size):
                valid = False
                message = f"{name} route segment {index} intersects {blocker_path}; costmap recovery will repel the robot."
                if strict:
                    raise RuntimeError(message)
                print(f"[COSTMAP]: {message}")
    return valid


def spawn_route_markers(name: str, route: list[tuple[float, float]], color: tuple[float, float, float]):
    for i, (x, y) in enumerate(route):
        spawn_cylinder(
            f"/World/Arena/{name}_RouteWaypoint_{i:02d}",
            radius=0.030,
            height=0.005,
            axis="Z",
            pos=(x, y, 0.010),
            color=color,
            opacity=0.75,
            semantic="behavior_route_waypoint",
        )


def spawn_base_armor(base_team: str, base_xy: tuple[float, float], color: tuple[float, float, float]):
    create_xform(f"/World/Arena/{base_team.capitalize()}BaseArmor")
    armor_height = 0.300
    z = armor_height * 0.5
    armor_thickness = 0.050
    armor_length = 0.250
    shield_color = (0.05, 0.22, 0.78)

    if base_team == "blue":
        # Four grounded armor plates segment the two open edges of the
        # 50cm base square: right edge plates 1/3 and lower edge plates 2/4.
        # This matches the national-rule numbering diagram and the archived
        # final_training_replay_overview reference.
        specs = [
            ("armor_1", (-1.025, 1.375, z), (armor_thickness, armor_length, armor_height)),
            ("armor_2", (-1.375, 1.025, z), (armor_length, armor_thickness, armor_height)),
            ("armor_3", (-1.025, 1.125, z), (armor_thickness, armor_length, armor_height)),
            ("armor_4", (-1.125, 1.025, z), (armor_length, armor_thickness, armor_height)),
        ]
    else:
        # Yellow mirrors the same base-edge armor layout.
        specs = [
            ("armor_1", (1.025, -1.375, z), (armor_thickness, armor_length, armor_height)),
            ("armor_2", (1.375, -1.025, z), (armor_length, armor_thickness, armor_height)),
            ("armor_3", (1.025, -1.125, z), (armor_thickness, armor_length, armor_height)),
            ("armor_4", (1.125, -1.025, z), (armor_length, armor_thickness, armor_height)),
        ]

    BASE_ARMOR[base_team] = []
    for index, (name, pos, size) in enumerate(specs):
        armor_path = f"/World/Arena/{base_team.capitalize()}BaseArmor/{name}"
        spawn_box(
            armor_path,
            size,
            pos,
            shield_color,
            collision=True,
            raycast=True,
            semantic=f"{base_team}_base_armor_{index + 1}",
        )
        register_nav_blocker(armor_path, pos, size)
        BASE_ARMOR[base_team].append(armor_path)


def target_path_from_name(name: str) -> str:
    return f"/World/Targets/{name}"
