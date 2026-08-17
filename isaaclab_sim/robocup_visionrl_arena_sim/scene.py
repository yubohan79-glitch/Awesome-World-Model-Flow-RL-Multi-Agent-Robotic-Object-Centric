from __future__ import annotations

import math


from ._bootstrap import (
    ARENA_SIZE,
    BASE_LINK_Z,
    BLUE_BASE_TARGET_XY,
    BLUE_BASE_TARGET_YAW,
    BLUE_BASE_XY,
    BLUE_ROBOT_PATH,
    BLUE_ROUTE,
    BLUE_START_XY,
    CAMERA_POSE,
    Camera,
    CameraCfg,
    DEMO_FLOW_FIRE_EVENTS,
    DEMO_FLOW_PATH_CACHE,
    DEMO_FLOW_POSES,
    DEMO_FLOW_RECOVERY_WINDOWS,
    DEMO_FLOW_TRIGGERED_EVENTS,
    DEPTH_CAMERA_POSE,
    Gf,
    IMU_POSE,
    LAST_FIRE_TIME,
    LIDAR_POSE,
    MATCH_CONTROLLERS,
    MATCH_STATE,
    NORTH_MIDDLE_TARGET_X,
    OBSTACLE_SIZE,
    PRIMARY_ROBOT_PATH,
    PUSHABLE_OBSTACLE_STARTS,
    RAYCAST_BOXES,
    ROBOT_BODY_HEIGHT,
    ROBOT_COLLISION_RADIUS,
    ROBOT_LENGTH,
    ROBOT_WIDTH,
    RayCaster,
    RayCasterCfg,
    SHOOTER_POSE,
    SIDE_GATE_TARGET_Y,
    SOUTH_MIDDLE_TARGET_X,
    TARGET_REGISTRY,
    TARGET_WALL_INSET,
    TOF_FRONT_POSE,
    UsdGeom,
    WALL_HEIGHT,
    WALL_THICKNESS,
    WHEEL_RADIUS,
    WHEEL_WIDTH,
    YELLOW_BASE_TARGET_XY,
    YELLOW_BASE_TARGET_YAW,
    YELLOW_BASE_XY,
    YELLOW_ROBOT_PATH,
    YELLOW_ROUTE,
    YELLOW_START_XY,
    ZONE_SIZE,
    args_cli,
    get_current_stage,
    patterns,
    sim_utils
)
from .controllers import initialize_demo_flow_controllers, initialize_match_controllers
from .costmap import apply_costmap_recovery, plan_safe_path
from .laser import scripted_fire_after_dwell
from .replay import (
    apply_replay_box_positions,
    apply_trained_replay_events,
    replay_row_at,
    trained_replay_pushable_pose
)
from .rules import inward_45deg_target_yaws
from .spawn import (
    rigid_physics_material,
    spawn_base_armor,
    spawn_box,
    spawn_cylinder,
    spawn_nav_blocker,
    spawn_pushable_obstacle,
    spawn_route_markers,
    spawn_target,
    target_path_from_name,
    validate_route
)
from .transforms import (
    create_xform,
    quat_from_euler,
    quat_rotate,
    set_xform
)

def design_arena():
    create_xform("/World/Arena")
    create_xform("/World/Targets")
    create_xform("/World/Targets/Fallen")
    create_xform("/World/Targets/Falling")
    create_xform("/World/Light")

    dome_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.86, 0.92))
    dome_cfg.func("/World/Light/Dome", dome_cfg)
    distant_cfg = sim_utils.DistantLightCfg(intensity=2600.0, color=(1.0, 0.96, 0.88), angle=0.65)
    distant_cfg.func("/World/Light/Main", distant_cfg, translation=(1.5, -2.0, 4.0), orientation=quat_from_euler(-0.8, 0.3, 0.35))

    spawn_box(
        "/World/Arena/Floor",
        (ARENA_SIZE, ARENA_SIZE, 0.02),
        (0.0, 0.0, -0.01),
        (0.15, 0.17, 0.16),
        collision=True,
        raycast=True,
        semantic="arena_floor",
        physics_material=rigid_physics_material(0.92, 0.74),
        contact_offset=0.004,
        rest_offset=0.0,
    )

    for idx, p in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0]):
        spawn_box(f"/World/Arena/GridX_{idx}", (0.008, ARENA_SIZE, 0.004), (p, 0.0, 0.004), (0.32, 0.34, 0.32))
        spawn_box(f"/World/Arena/GridY_{idx}", (ARENA_SIZE, 0.008, 0.004), (0.0, p, 0.004), (0.32, 0.34, 0.32))

    zones = [
        ("BlueBase", BLUE_BASE_XY, (0.08, 0.25, 0.72), "blue_base_zone"),
        ("BlueStart", BLUE_START_XY, (0.12, 0.36, 0.90), "blue_start_zone"),
        ("YellowStart", YELLOW_START_XY, (0.95, 0.86, 0.08), "yellow_start_zone"),
        ("YellowBase", YELLOW_BASE_XY, (0.88, 0.78, 0.06), "yellow_base_zone"),
    ]
    for name, (x, y), color, semantic in zones:
        spawn_box(
            f"/World/Arena/{name}",
            (ZONE_SIZE, ZONE_SIZE, 0.006),
            (x, y, 0.006),
            color,
            semantic=semantic,
            opacity=0.86,
        )

    wall_color = (0.76, 0.78, 0.72)
    wall_z = WALL_HEIGHT * 0.5
    wall_span = ARENA_SIZE + WALL_THICKNESS * 2.0
    spawn_nav_blocker(
        "/World/Arena/WallWest",
        (WALL_THICKNESS, wall_span, WALL_HEIGHT),
        (-(ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), 0.0, wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallEast",
        (WALL_THICKNESS, wall_span, WALL_HEIGHT),
        ((ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), 0.0, wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallSouth",
        (wall_span, WALL_THICKNESS, WALL_HEIGHT),
        (0.0, -(ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), wall_z),
        wall_color,
        semantic="arena_wall",
    )
    spawn_nav_blocker(
        "/World/Arena/WallNorth",
        (wall_span, WALL_THICKNESS, WALL_HEIGHT),
        (0.0, (ARENA_SIZE * 0.5 + WALL_THICKNESS * 0.5), wall_z),
        wall_color,
        semantic="arena_wall",
    )

    internal_specs = [
        ("MidWallWest", (-1.00, 0.0, wall_z), (1.00, WALL_THICKNESS, WALL_HEIGHT)),
        ("MidWallEast", (1.00, 0.0, wall_z), (1.00, WALL_THICKNESS, WALL_HEIGHT)),
        ("BlueStartEastRail", (0.00, 1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
        ("YellowStartWestRail", (0.00, -1.25, wall_z), (WALL_THICKNESS, 0.50, WALL_HEIGHT)),
    ]
    for name, pos, size in internal_specs:
        spawn_nav_blocker(
            f"/World/Arena/{name}",
            size,
            pos,
            (0.70, 0.72, 0.67),
            semantic="internal_wall",
        )

    obstacles = [
        ("RandomObstacleNorthEast", (*PUSHABLE_OBSTACLE_STARTS["box_ne"], OBSTACLE_SIZE * 0.5)),
        ("RandomObstacleSouthWest", (*PUSHABLE_OBSTACLE_STARTS["box_sw"], OBSTACLE_SIZE * 0.5)),
    ]
    for name, pos in obstacles:
        spawn_pushable_obstacle(
            f"/World/Arena/{name}",
            (OBSTACLE_SIZE, OBSTACLE_SIZE, OBSTACLE_SIZE),
            pos,
            (0.92, 0.05, 0.02),
            semantic="random_obstacle_30cm",
        )

    target_edge = ARENA_SIZE * 0.5 - TARGET_WALL_INSET
    target_yaws = inward_45deg_target_yaws()
    normal_targets = [
        ("T01_NorthMiddle", (NORTH_MIDDLE_TARGET_X, target_edge), target_yaws["T01_NorthMiddle"]),
        ("T02_NorthEast", (target_edge, target_edge), target_yaws["T02_NorthEast"]),
        ("T03_WestAboveGate", (-target_edge, SIDE_GATE_TARGET_Y), target_yaws["T03_WestAboveGate"]),
        ("T04_WestBelowGate", (-target_edge, -SIDE_GATE_TARGET_Y), target_yaws["T04_WestBelowGate"]),
        ("T05_EastAboveGate", (target_edge, SIDE_GATE_TARGET_Y), target_yaws["T05_EastAboveGate"]),
        ("T06_EastBelowGate", (target_edge, -SIDE_GATE_TARGET_Y), target_yaws["T06_EastBelowGate"]),
        ("T07_SouthWest", (-target_edge, -target_edge), target_yaws["T07_SouthWest"]),
        ("T08_SouthMiddle", (SOUTH_MIDDLE_TARGET_X, -target_edge), target_yaws["T08_SouthMiddle"]),
    ]
    for name, xy, yaw in normal_targets:
        spawn_target(f"/World/Targets/{name}", xy, yaw, tag_id=1, frame_color=(0.25, 0.26, 0.25))

    spawn_target(
        "/World/Targets/BlueBaseTarget",
        BLUE_BASE_TARGET_XY,
        BLUE_BASE_TARGET_YAW,
        tag_id=3,
        frame_color=(0.08, 0.20, 0.56),
        base_target=True,
    )
    spawn_target(
        "/World/Targets/YellowBaseTarget",
        YELLOW_BASE_TARGET_XY,
        YELLOW_BASE_TARGET_YAW,
        tag_id=2,
        frame_color=(0.64, 0.48, 0.10),
        base_target=True,
    )
    spawn_base_armor("blue", BLUE_BASE_XY, (0.10, 0.34, 0.90))
    spawn_base_armor("yellow", YELLOW_BASE_XY, (0.90, 0.72, 0.12))

    validate_route("yellow", YELLOW_ROUTE)
    validate_route("blue", BLUE_ROUTE)
    spawn_route_markers("Yellow", YELLOW_ROUTE, (0.95, 0.86, 0.08))
    spawn_route_markers("Blue", BLUE_ROUTE, (0.12, 0.36, 0.90))


def design_robot(
    robot_path: str,
    start_xy: tuple[float, float],
    start_yaw: float,
    team_color: tuple[float, float, float],
    accent_color: tuple[float, float, float],
    beam_color: tuple[float, float, float],
):
    start_pose = (start_xy[0], start_xy[1], 0.0)
    create_xform(robot_path, translation=start_pose, orientation=quat_from_euler(0.0, 0.0, start_yaw))

    body_center_z = BASE_LINK_Z + ROBOT_BODY_HEIGHT * 0.5
    spawn_box(
        f"{robot_path}/base_link",
        (ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_BODY_HEIGHT),
        (0.0, 0.0, body_center_z),
        (0.025, 0.028, 0.030),
        collision=True,
        semantic="robot_base_link",
    )
    spawn_box(
        f"{robot_path}/collision_hull",
        (ROBOT_LENGTH, ROBOT_WIDTH, ROBOT_BODY_HEIGHT * 0.82),
        (0.0, 0.0, body_center_z),
        team_color,
        collision=True,
        semantic="robot_collision_hull",
        opacity=0.18,
        rigid_body=True,
        kinematic=True,
        mass=8.0,
        disable_gravity=True,
    )
    spawn_box(
        f"{robot_path}/top_plate",
        (0.28, 0.18, 0.012),
        (0.0, 0.0, BASE_LINK_Z + ROBOT_BODY_HEIGHT + 0.018),
        team_color,
        semantic="robot_top_plate",
    )
    spawn_box(
        f"{robot_path}/front_bumper",
        (0.035, ROBOT_WIDTH + 0.03, 0.05),
        (ROBOT_LENGTH * 0.5 + 0.010, 0.0, BASE_LINK_Z + 0.045),
        accent_color,
        collision=True,
        semantic="robot_bumper",
    )
    spawn_box(
        f"{robot_path}/battery",
        (0.11, 0.10, 0.045),
        (-0.06, 0.0, BASE_LINK_Z + ROBOT_BODY_HEIGHT + 0.050),
        (0.16, 0.18, 0.18),
        semantic="battery_pack",
    )
    spawn_box(
        f"{robot_path}/imu_link",
        (0.044, 0.034, 0.014),
        IMU_POSE,
        (0.12, 0.72, 0.40),
        semantic="imu_9axis_module",
    )
    spawn_box(
        f"{robot_path}/imu_axis_x",
        (0.050, 0.004, 0.004),
        (IMU_POSE[0] + 0.025, IMU_POSE[1], IMU_POSE[2] + 0.012),
        (0.90, 0.05, 0.05),
        semantic="imu_x_axis",
    )
    spawn_box(
        f"{robot_path}/imu_axis_y",
        (0.004, 0.050, 0.004),
        (IMU_POSE[0], IMU_POSE[1] + 0.025, IMU_POSE[2] + 0.012),
        (0.05, 0.75, 0.08),
        semantic="imu_y_axis",
    )

    left_y = ROBOT_WIDTH * 0.5 + WHEEL_WIDTH * 0.5
    right_y = -left_y
    for name, y in [("left_wheel_link", left_y), ("right_wheel_link", right_y)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=WHEEL_RADIUS,
            height=WHEEL_WIDTH,
            axis="Y",
            pos=(-0.03, y, WHEEL_RADIUS),
            color=(0.015, 0.015, 0.014),
            collision=True,
            semantic=name,
        )
        spawn_cylinder(
            f"{robot_path}/{name}_hub",
            radius=WHEEL_RADIUS * 0.55,
            height=WHEEL_WIDTH + 0.004,
            axis="Y",
            pos=(-0.03, y, WHEEL_RADIUS),
            color=(0.56, 0.58, 0.55),
            semantic="wheel_hub",
        )
        spawn_box(
            f"{robot_path}/{name}_stripe",
            (0.006, WHEEL_WIDTH + 0.006, WHEEL_RADIUS * 1.6),
            (-0.03 + WHEEL_RADIUS * 0.45, y, WHEEL_RADIUS),
            (0.92, 0.92, 0.86),
            semantic="wheel_rotation_mark",
        )

    for name, x in [("front_caster", 0.125), ("rear_caster", -0.145)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=0.020,
            height=0.018,
            axis="Z",
            pos=(x, 0.0, 0.020),
            color=(0.09, 0.09, 0.085),
            collision=True,
            semantic=name,
        )

    spawn_cylinder(
        f"{robot_path}/laser_link",
        radius=0.035,
        height=0.035,
        axis="Z",
        pos=LIDAR_POSE,
        color=accent_color,
        collision=True,
        semantic="rplidar_frame",
    )
    spawn_cylinder(
        f"{robot_path}/lidar_cap",
        radius=0.030,
        height=0.008,
        axis="Z",
        pos=(LIDAR_POSE[0], LIDAR_POSE[1], LIDAR_POSE[2] + 0.022),
        color=(0.02, 0.02, 0.024),
        semantic="lidar_rotor",
    )
    spawn_box(
        f"{robot_path}/camera_link",
        (0.040, 0.030, 0.030),
        CAMERA_POSE,
        (0.015, 0.016, 0.018),
        collision=True,
        semantic="rgb_camera",
    )
    spawn_box(
        f"{robot_path}/depth_camera_link",
        (0.034, 0.024, 0.024),
        DEPTH_CAMERA_POSE,
        (0.025, 0.025, 0.030),
        collision=True,
        semantic="depth_camera",
    )
    spawn_cylinder(
        f"{robot_path}/camera_lens",
        radius=0.012,
        height=0.010,
        axis="X",
        pos=(CAMERA_POSE[0] + 0.025, CAMERA_POSE[1], CAMERA_POSE[2]),
        color=(0.03, 0.05, 0.07),
        semantic="camera_lens",
    )
    spawn_cylinder(
        f"{robot_path}/depth_camera_lens",
        radius=0.009,
        height=0.009,
        axis="X",
        pos=(DEPTH_CAMERA_POSE[0] + 0.023, DEPTH_CAMERA_POSE[1], DEPTH_CAMERA_POSE[2]),
        color=(0.02, 0.04, 0.07),
        semantic="depth_camera_lens",
    )
    for name, y in [("front_tof_left", 0.070), ("front_tof_right", -0.070)]:
        spawn_box(
            f"{robot_path}/{name}",
            (0.018, 0.018, 0.014),
            (TOF_FRONT_POSE[0], y, TOF_FRONT_POSE[2]),
            (0.04, 0.30, 0.36),
            semantic="tof_range_sensor",
        )
    for name, y in [("front_bumper_contact_left", 0.075), ("front_bumper_contact_right", -0.075)]:
        spawn_box(
            f"{robot_path}/{name}",
            (0.012, 0.055, 0.030),
            (ROBOT_LENGTH * 0.5 + 0.033, y, BASE_LINK_Z + 0.047),
            (0.86, 0.18, 0.10),
            semantic="bumper_contact_sensor",
        )
    for name, y in [("left_wheel_encoder", left_y), ("right_wheel_encoder", right_y)]:
        spawn_cylinder(
            f"{robot_path}/{name}",
            radius=0.018,
            height=0.006,
            axis="Y",
            pos=(-0.03, y * 0.90, WHEEL_RADIUS),
            color=(0.15, 0.12, 0.08),
            semantic="wheel_encoder",
        )
    spawn_cylinder(
        f"{robot_path}/shooter_link",
        radius=0.008,
        height=0.080,
        axis="X",
        pos=SHOOTER_POSE,
        color=beam_color,
        emissive=tuple(channel * 0.45 for channel in beam_color),
        collision=True,
        semantic="fixed_laser_shooter",
    )
    spawn_box(
        f"{robot_path}/laser_beam_preview",
        (1.05, 0.006, 0.006),
        (SHOOTER_POSE[0] + 0.55, SHOOTER_POSE[1], SHOOTER_POSE[2]),
        beam_color,
        opacity=0.34,
        emissive=beam_color,
        semantic="low_power_fixed_laser_beam",
    )

    # ROS/TF style sensor frames used by IsaacLab sensors.
    create_xform(f"{robot_path}/CameraFrame", translation=CAMERA_POSE)
    create_xform(f"{robot_path}/LidarFrame", translation=LIDAR_POSE)
    create_xform(f"{robot_path}/ImuFrame", translation=IMU_POSE)
    create_xform(f"{robot_path}/DepthCameraFrame", translation=DEPTH_CAMERA_POSE)

    # Frame axes at base_link: x red, y green, z blue.
    spawn_box(f"{robot_path}/tf_x_axis", (0.18, 0.008, 0.008), (0.09, 0.0, BASE_LINK_Z), (0.90, 0.05, 0.05), emissive=(0.25, 0.0, 0.0))
    spawn_box(f"{robot_path}/tf_y_axis", (0.008, 0.18, 0.008), (0.0, 0.09, BASE_LINK_Z), (0.05, 0.70, 0.08), emissive=(0.0, 0.20, 0.0))
    spawn_box(f"{robot_path}/tf_z_axis", (0.008, 0.008, 0.16), (0.0, 0.0, BASE_LINK_Z + 0.08), (0.08, 0.18, 0.90), emissive=(0.0, 0.0, 0.28))


def create_lidar_proxy_mesh() -> str:
    """Create a single static mesh for the IsaacLab RayCaster.

    This local IsaacLab build only supports one static mesh in RayCaster, so
    the field collision boxes are mirrored into one invisible mesh. The visible
    scene remains made from separate physical parts.
    """
    stage = get_current_stage()
    proxy_root = "/World/LidarProxy"
    proxy_mesh_path = f"{proxy_root}/static_arena_mesh"
    create_xform(proxy_root)

    vertices: list[Gf.Vec3f] = []
    indices: list[int] = []
    counts: list[int] = []

    corner_signs = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ]
    tri_faces = [
        (0, 2, 3),
        (0, 3, 1),
        (4, 5, 7),
        (4, 7, 6),
        (0, 1, 5),
        (0, 5, 4),
        (2, 6, 7),
        (2, 7, 3),
        (0, 4, 6),
        (0, 6, 2),
        (1, 3, 7),
        (1, 7, 5),
    ]

    for size, center, quat in RAYCAST_BOXES:
        base = len(vertices)
        half = (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5)
        for sx, sy, sz in corner_signs:
            rotated = quat_rotate(quat, (sx * half[0], sy * half[1], sz * half[2]))
            vertices.append(Gf.Vec3f(center[0] + rotated[0], center[1] + rotated[1], center[2] + rotated[2]))
        for a, b, c in tri_faces:
            indices.extend([base + a, base + b, base + c])
            counts.append(3)

    mesh = UsdGeom.Mesh.Define(stage, proxy_mesh_path)
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    return proxy_root


def create_sensor_streams() -> dict[str, object]:
    if args_cli.no_sensor_streams or not args_cli.enable_sensor_streams:
        return {}

    camera_cfg = CameraCfg(
        prim_path=f"{PRIMARY_ROBOT_PATH}/CameraFrame/CameraSensor",
        update_period=1.0 / 15.0,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.6,
            focus_distance=2.0,
            horizontal_aperture=4.8,
            clipping_range=(0.05, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        semantic_segmentation_mapping={
            "class:arena_wall": (190, 190, 170, 255),
            "class:random_obstacle_30cm": (235, 132, 44, 255),
            "class:robot_base_link": (50, 50, 54, 255),
            "class:apriltag_visual": (20, 20, 20, 255),
            "class:arena_floor": (72, 78, 73, 255),
        },
    )
    camera = Camera(camera_cfg)

    lidar_proxy_path = create_lidar_proxy_mesh()
    lidar_cfg = RayCasterCfg(
        prim_path=f"{PRIMARY_ROBOT_PATH}/LidarFrame",
        update_period=1.0 / 12.0,
        mesh_prim_paths=[lidar_proxy_path],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-1.0, 1.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=4.0,
        debug_vis=not args_cli.headless,
    )
    lidar = RayCaster(lidar_cfg)
    sensors: dict[str, object] = {"camera": camera, "lidar": lidar}
    try:
        from isaaclab.sensors.imu import Imu, ImuCfg

        imu_cfg = ImuCfg(
            prim_path=f"{PRIMARY_ROBOT_PATH}/ImuFrame",
            update_period=1.0 / 100.0,
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            gravity_bias=(0.0, 0.0, 9.81),
        )
        sensors["imu"] = Imu(imu_cfg)
    except Exception as exc:
        print(f"[WARN]: IsaacLab IMU stream unavailable ({exc}); visual/ROS IMU model is still present.")
    return sensors


def route_pose(
    t: float,
    route: list[tuple[float, float]],
    *,
    speed: float = 0.22,
) -> tuple[tuple[float, float, float], float]:
    segment_lengths = [
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        return (route[0][0], route[0][1], 0.0), 0.0

    travel = (t * speed) % (total_length * 2.0)
    reverse = travel > total_length
    distance = total_length * 2.0 - travel if reverse else travel

    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            local = 0.0 if length <= 1e-9 else (distance - walked) / length
            eased = 0.5 - 0.5 * math.cos(max(0.0, min(1.0, local)) * math.pi)
            x0, y0 = route[index]
            x1, y1 = route[index + 1]
            x = x0 + (x1 - x0) * eased
            y = y0 + (y1 - y0) * eased
            yaw = math.atan2(y1 - y0, x1 - x0)
            if reverse:
                yaw += math.pi
            return (x, y, 0.0), yaw
        walked += length

    x, y = route[-1]
    return (x, y, 0.0), 0.0


def target_xy_for_name(target_name: str) -> tuple[float, float] | None:
    target = TARGET_REGISTRY.get(target_path_from_name(target_name))
    if target is None:
        return None
    xy = target["xy"]
    assert isinstance(xy, tuple)
    return xy


def finite_path_pose(
    path: list[tuple[float, float]],
    progress: float,
    fallback_yaw: float,
) -> tuple[tuple[float, float, float], float]:
    if len(path) < 2:
        x, y = path[0]
        return (x, y, 0.0), fallback_yaw

    segment_lengths = [
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    total_length = sum(segment_lengths)
    if total_length <= 1e-9:
        x, y = path[-1]
        return (x, y, 0.0), fallback_yaw

    distance = max(0.0, min(1.0, progress)) * total_length
    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            local = 0.0 if length <= 1e-9 else (distance - walked) / length
            x0, y0 = path[index]
            x1, y1 = path[index + 1]
            x = x0 + (x1 - x0) * local
            y = y0 + (y1 - y0) * local
            yaw = math.atan2(y1 - y0, x1 - x0) if length > 1e-9 else fallback_yaw
            return (x, y, 0.0), yaw

    x, y = path[-1]
    return (x, y, 0.0), fallback_yaw


def demo_segment_path(team: str, segment_index: int, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    key = (team, segment_index)
    cached = DEMO_FLOW_PATH_CACHE.get(key)
    if cached is not None:
        return cached
    path = plan_safe_path(start_xy, goal_xy)
    validate_route(f"demo_{team}_{segment_index}", path)
    DEMO_FLOW_PATH_CACHE[key] = path
    return path


def demo_flow_pose(team: str, t: float) -> tuple[tuple[float, float, float], float]:
    script = DEMO_FLOW_POSES[team]
    if t <= script[0][0]:
        x, y = script[0][1]
        start_yaw = math.pi * 0.5 if team == "yellow" else -math.pi * 0.5
        return (x, y, 0.0), start_yaw

    for index in range(len(script) - 1):
        t0, xy0, look0 = script[index]
        t1, xy1, look1 = script[index + 1]
        if t <= t1:
            alpha = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
            path = demo_segment_path(team, index, xy0, xy1)
            (x, y, _z), travel_yaw = finite_path_pose(path, eased, math.pi * 0.5 if team == "yellow" else -math.pi * 0.5)
            (x, y), _costmap_touch, _hard_touch = apply_costmap_recovery((x, y), f"demo_{team}")
            look_target = look1 or look0
            target_xy = target_xy_for_name(look_target) if look_target else None
            if target_xy is not None:
                yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
            else:
                yaw = travel_yaw
            for start, end in DEMO_FLOW_RECOVERY_WINDOWS:
                if start <= t <= end:
                    spin_sign = 1.0 if team == "yellow" else -1.0
                    yaw += spin_sign * (t - start) * 2.8
            return (x, y, 0.0), yaw

    x, y = script[-1][1]
    look_target = script[-1][2]
    target_xy = target_xy_for_name(look_target) if look_target else None
    if target_xy is not None:
        yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
    else:
        yaw = math.pi * 0.5 if team == "yellow" else -math.pi * 0.5
    return (x, y, 0.0), yaw


def trigger_demo_flow_events(t: float):
    for event_index, (event_time, team, target_name) in enumerate(DEMO_FLOW_FIRE_EVENTS):
        if event_index in DEMO_FLOW_TRIGGERED_EVENTS or t < event_time:
            continue
        if MATCH_STATE["winner"] is not None:
            DEMO_FLOW_TRIGGERED_EVENTS.add(event_index)
            continue
        target_path = target_path_from_name(target_name)
        if target_path not in TARGET_REGISTRY:
            raise RuntimeError(f"Demo flow target not found: {target_name}")
        LAST_FIRE_TIME[team] = t
        scripted_fire_after_dwell(team, target_path)
        DEMO_FLOW_TRIGGERED_EVENTS.add(event_index)


def update_demo_flow_animation(t: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    initialize_demo_flow_controllers()
    yellow_pose = demo_flow_pose("yellow", t)
    blue_pose = demo_flow_pose("blue", t)
    yellow_pose, blue_pose = resolve_robot_contact(yellow_pose, blue_pose, t)
    MATCH_CONTROLLERS["yellow"].set_pose(yellow_pose)
    MATCH_CONTROLLERS["blue"].set_pose(blue_pose)
    poses = {"yellow": yellow_pose, "blue": blue_pose}

    for robot_path, team in ((YELLOW_ROBOT_PATH, "yellow"), (BLUE_ROBOT_PATH, "blue")):
        controller = MATCH_CONTROLLERS[team]
        spin_sign = 1.0 if team == "yellow" else -1.0
        controller.left_wheel_spin = spin_sign * t * 8.0
        controller.right_wheel_spin = spin_sign * t * 8.4
        pos, yaw = poses[team]
        set_xform(robot_path, pos, quat_from_euler(0.0, 0.0, yaw))
        update_robot_parts(robot_path, team, t)
    trigger_demo_flow_events(t)
    return poses


def update_robot_parts(robot_path: str, team: str, t: float):
    controller = MATCH_CONTROLLERS.get(team)
    left_spin = controller.left_wheel_spin if controller else t * 7.0
    right_spin = controller.right_wheel_spin if controller else t * 7.0
    left_y = ROBOT_WIDTH * 0.5 + WHEEL_WIDTH * 0.5
    right_y = -left_y
    for side_y, side_name, wheel_spin in (
        (left_y, "left_wheel_link", left_spin),
        (right_y, "right_wheel_link", right_spin),
    ):
        set_xform(f"{robot_path}/{side_name}", (-0.03, side_y, WHEEL_RADIUS), quat_from_euler(0.0, wheel_spin, 0.0))
        set_xform(
            f"{robot_path}/{side_name}_hub",
            (-0.03, side_y, WHEEL_RADIUS),
            quat_from_euler(0.0, wheel_spin, 0.0),
        )
    set_xform(
        f"{robot_path}/lidar_cap",
        (LIDAR_POSE[0], LIDAR_POSE[1], LIDAR_POSE[2] + 0.022),
        quat_from_euler(0.0, 0.0, t * 12.0),
    )


def resolve_robot_contact(
    yellow_pose: tuple[tuple[float, float, float], float],
    blue_pose: tuple[tuple[float, float, float], float],
    t: float,
) -> tuple[tuple[tuple[float, float, float], float], tuple[tuple[float, float, float], float]]:
    yellow_pos, yellow_yaw = yellow_pose
    blue_pos, blue_yaw = blue_pose
    dx = blue_pos[0] - yellow_pos[0]
    dy = blue_pos[1] - yellow_pos[1]
    distance = math.hypot(dx, dy)
    min_distance = ROBOT_COLLISION_RADIUS * 2.0
    if distance >= min_distance:
        MATCH_STATE["robot_contact"] = False
        return yellow_pose, blue_pose

    if distance < 1e-6:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = dx / distance, dy / distance
    push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
    yellow_pos = (yellow_pos[0] - nx * push, yellow_pos[1] - ny * push, yellow_pos[2])
    blue_pos = (blue_pos[0] + nx * push, blue_pos[1] + ny * push, blue_pos[2])
    MATCH_STATE["robot_contact"] = True
    if "yellow" in MATCH_CONTROLLERS:
        MATCH_CONTROLLERS["yellow"].notify_robot_contact(t)
    if "blue" in MATCH_CONTROLLERS:
        MATCH_CONTROLLERS["blue"].notify_robot_contact(t)
    if t - float(MATCH_STATE["last_contact_print"]) > 2.0:
        MATCH_STATE["last_contact_print"] = t
        print("[RULE]: Robot contact resolved: yellow and blue collision hulls separated.")
    return (yellow_pos, yellow_yaw), (blue_pos, blue_yaw)


def update_trained_replay_animation(t: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    yellow_row = replay_row_at("yellow", t)
    blue_row = replay_row_at("blue", t)
    apply_replay_box_positions(yellow_row)
    yellow_pos = (float(yellow_row["x"]), float(yellow_row["y"]), 0.0)
    blue_pos = (float(blue_row["x"]), float(blue_row["y"]), 0.0)
    yellow_pos = trained_replay_pushable_pose("yellow", yellow_pos, float(yellow_row["yaw"]))
    blue_pos = trained_replay_pushable_pose("blue", blue_pos, float(blue_row["yaw"]))

    poses = {
        "yellow": (yellow_pos, float(yellow_row["yaw"])),
        "blue": (blue_pos, float(blue_row["yaw"])),
    }

    for robot_path, team in ((YELLOW_ROBOT_PATH, "yellow"), (BLUE_ROBOT_PATH, "blue")):
        pos, yaw = poses[team]
        set_xform(robot_path, pos, quat_from_euler(0.0, 0.0, yaw))
        controller = MATCH_CONTROLLERS.get(team)
        if controller is not None:
            controller.set_pose((pos, yaw))
        update_robot_parts(robot_path, team, t)

    apply_trained_replay_events(t)
    return poses


def update_robot_animation(t: float) -> dict[str, tuple[tuple[float, float, float], float]]:
    if args_cli.replay_trace:
        return update_trained_replay_animation(t)

    if args_cli.demo_flow:
        return update_demo_flow_animation(t)

    if args_cli.static_robot:
        poses = {
            "yellow": ((YELLOW_START_XY[0], YELLOW_START_XY[1], 0.0), math.pi * 0.5),
            "blue": ((BLUE_START_XY[0], BLUE_START_XY[1], 0.0), -math.pi * 0.5),
        }
    else:
        initialize_match_controllers()
        yellow_pose = MATCH_CONTROLLERS["yellow"].update(t)
        blue_pose = MATCH_CONTROLLERS["blue"].update(t)
        yellow_pose, blue_pose = resolve_robot_contact(yellow_pose, blue_pose, t)
        MATCH_CONTROLLERS["yellow"].set_pose(yellow_pose)
        MATCH_CONTROLLERS["blue"].set_pose(blue_pose)
        poses = {"yellow": yellow_pose, "blue": blue_pose}

    for robot_path, team in ((YELLOW_ROBOT_PATH, "yellow"), (BLUE_ROBOT_PATH, "blue")):
        pos, yaw = poses[team]
        set_xform(robot_path, pos, quat_from_euler(0.0, 0.0, yaw))
        update_robot_parts(robot_path, team, t)
    return poses
