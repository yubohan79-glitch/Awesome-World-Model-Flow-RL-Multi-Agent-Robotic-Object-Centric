from __future__ import annotations

"""IsaacLab scene for the RoboCup VisionRL portfolio project.

Run from the local IsaacLab checkout:

    isaaclab.bat -p <this_file.py> --enable_cameras

The scene is metric and keeps the robot/sensor dimensions aligned with the
ROS2 description in rcvrl_description/urdf/robocup_visionrl_robot.urdf.xacro.
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RoboCup VisionRL IsaacLab simulation scene.")
parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until the GUI closes.")
parser.add_argument("--seed", type=int, default=7, help="Deterministic layout seed for sim2real preview variation.")
parser.add_argument("--static_robot", action="store_true", help="Keep both robots at their start zones.")
parser.add_argument(
    "--demo_flow",
    action="store_true",
    help="Run a deterministic full-match portfolio replay with target falls, armor removal, recovery, and base hit.",
)
parser.add_argument("--replay_trace", type=str, default="", help="CSV trace produced by replay_policy_strict.py.")
parser.add_argument("--replay_events", type=str, default="", help="JSONL events produced by replay_policy_strict.py.")
parser.add_argument("--replay_episode", type=int, default=0, help="Episode index to replay from --replay_trace.")
parser.add_argument("--record_video", type=str, default="", help="Optional MP4 output path recorded from an IsaacLab RGB camera.")
parser.add_argument(
    "--record_view",
    choices=["overview", "top", "yellow_pov", "blue_pov"],
    default="overview",
    help="Camera view for --record_video.",
)
parser.add_argument("--record_fps", type=int, default=30, help="Video frame rate for --record_video.")
parser.add_argument("--record_width", type=int, default=1600, help="Video width for --record_video.")
parser.add_argument("--record_height", type=int, default=900, help="Video height for --record_video.")
parser.add_argument(
    "--enable_sensor_streams",
    action="store_true",
    help="Start live IsaacLab camera/lidar sensors. Off by default to keep GUI preview and shutdown stable.",
)
parser.add_argument("--no_sensor_streams", action="store_true", help=argparse.SUPPRESS)
parser.add_argument(
    "--save_usd",
    type=str,
    default="",
    help="Optional USD export path. Defaults to isaaclab_sim/output/robocup_visionrl_arena.usd.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Live camera streams require Replicator. Keep them opt-in because this PC's
# Isaac Sim 5.1 build can hang during headless shutdown after semantic camera use.
if (args_cli.enable_sensor_streams or args_cli.record_video) and not args_cli.no_sensor_streams:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



if args_cli.record_video:
    pass


ARENA_SIZE = 3.0
WALL_HEIGHT = 0.50
WALL_THICKNESS = 0.04
ZONE_SIZE = 0.50
OBSTACLE_SIZE = 0.30
TAG_SIZE = 0.05
TAG_BOTTOM_Z = 0.07
TAG_CENTER_Z = TAG_BOTTOM_Z + TAG_SIZE * 0.5

ROBOT_LENGTH = 0.34
ROBOT_WIDTH = 0.24
ROBOT_BODY_HEIGHT = 0.16
ROBOT_TOTAL_HEIGHT = 0.245
WHEEL_RADIUS = 0.045
WHEEL_WIDTH = 0.025
BASE_LINK_Z = WHEEL_RADIUS
LIDAR_POSE = (0.06, 0.0, BASE_LINK_Z + 0.19)
CAMERA_POSE = (0.18, 0.0, BASE_LINK_Z + 0.18)
RECORDING_POV_CAMERA_POSE = (-0.04, 0.0, BASE_LINK_Z + 0.38)
SHOOTER_POSE = (0.20, 0.0, BASE_LINK_Z + 0.14)
SHOOTER_FORWARD_OFFSET = SHOOTER_POSE[0]
IMU_POSE = (-0.02, 0.0, BASE_LINK_Z + 0.11)
DEPTH_CAMERA_POSE = (0.165, 0.045, BASE_LINK_Z + 0.17)
TOF_FRONT_POSE = (0.185, 0.0, BASE_LINK_Z + 0.075)

COLLISION_PRIMS: list[str] = []
RAYCAST_BOXES: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]] = []
NAV_BLOCKERS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
LASER_BLOCKERS: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
PUSHABLE_OBSTACLES: dict[str, dict[str, object]] = {}
TARGET_REGISTRY: dict[str, dict[str, object]] = {}
BASE_ARMOR: dict[str, list[str]] = {"yellow": [], "blue": []}
LAST_FIRE_TIME: dict[str, float] = {"yellow": -99.0, "blue": -99.0}
LASER_LOCKS: dict[str, dict[str, float | str]] = {
    "yellow": {"target_path": "", "start_time": -99.0},
    "blue": {"target_path": "", "start_time": -99.0},
}
ARMOR_REMOVALS: list[dict[str, object]] = []
TARGET_FALLS: list[dict[str, object]] = []
MATCH_STATE: dict[str, object] = {
    "winner": None,
    "robot_contact": False,
    "last_contact_print": -99.0,
    "last_score_print": -99.0,
    "score_yellow": 0,
    "score_blue": 0,
    "current_time": 0.0,
    "last_event": "ready",
}
MATCH_CONTROLLERS: dict[str, object] = {}

YELLOW_ROBOT_PATH = "/World/RoboCupVisionRL_Yellow"
BLUE_ROBOT_PATH = "/World/RoboCupVisionRL_Blue"
PRIMARY_ROBOT_PATH = YELLOW_ROBOT_PATH

BLUE_BASE_XY = (-1.25, 1.25)
BLUE_START_XY = (-0.25, 1.25)
YELLOW_START_XY = (0.25, -1.25)
YELLOW_BASE_XY = (1.25, -1.25)
BLUE_BASE_TARGET_XY = (-1.36, 1.36)
YELLOW_BASE_TARGET_XY = (1.36, -1.36)
BLUE_BASE_TARGET_YAW = -math.pi / 4.0
YELLOW_BASE_TARGET_YAW = 3.0 * math.pi / 4.0
YELLOW_ATTACK_BLUE_BASE_XY = (-0.72, 1.32)
BLUE_ATTACK_YELLOW_BASE_XY = (0.72, -1.32)
YELLOW_DEMO_START_XY = (0.38, -1.18)
BLUE_DEMO_START_XY = (-0.38, 1.18)
ROUTE_CLEARANCE = ROBOT_WIDTH * 0.5 + 0.04
ROBOT_COLLISION_RADIUS = math.hypot(ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5)
ROBOT_PUSHABLE_CLEARANCE_RADIUS = ROBOT_COLLISION_RADIUS + 0.030
ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS = ROBOT_PUSHABLE_CLEARANCE_RADIUS + 0.065
ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS = (ROBOT_LENGTH * 0.5 + 0.110, ROBOT_WIDTH * 0.5 + WHEEL_WIDTH + 0.062)
NORMAL_SHOOT_MIN_RANGE = 0.05
NORMAL_SHOOT_RANGE = 0.50
NORMAL_SHOOT_IDEAL_DISTANCE = 0.30
BASE_SHOOT_MIN_RANGE = 0.20
BASE_SHOOT_RANGE = 0.80
BASE_SHOOT_IDEAL_DISTANCE = 0.48
SHOOT_MIN_RANGE = NORMAL_SHOOT_MIN_RANGE
SHOOT_RANGE = NORMAL_SHOOT_RANGE
SHOOT_HIT_RADIUS = 0.028
BASE_HIT_RADIUS = 0.018
SHOOT_IDEAL_DISTANCE = NORMAL_SHOOT_IDEAL_DISTANCE
FIRE_COOLDOWN = 1.4
LASER_DWELL_REQUIRED_S = 0.80
LASER_DWELL_FULL_CONFIDENCE_S = 2.00
TARGET_WALL_INSET = 0.240
TARGET_WALL_ANGLE_RAD = math.radians(45.0)
NORTH_MIDDLE_TARGET_X = 0.18
SOUTH_MIDDLE_TARGET_X = -0.18
SIDE_GATE_TARGET_Y = 0.24
BASE_HIT_SUCCESS_BY_NORMAL_HITS = {0: 0.0, 1: 0.40, 2: 0.55, 3: 0.80, 4: 0.95}
BASE_ARMOR_LIFT_CLEARANCE_Z = WALL_HEIGHT + 0.24
MATCH_DRIVE_SPEED = 0.72
MATCH_AIM_TIME = 0.45
MATCH_DURATION_S = 180.0
PLANNER_GRID_RESOLUTION = 0.10
BASE_RUSH_MIN_QUALITY = 0.34
BLOCK_HOLD_S = 4.0
BLOCK_LEAD_SCORE = 10
BLOCK_LATE_TIME_S = 45.0
LOCALIZATION_RECOVERY_THRESHOLD = 0.58
LOCALIZATION_RECOVERY_ROTATION_RAD = math.tau * 1.08
LOCALIZATION_CONTACT_LOSS = 0.42
LOCALIZATION_STUCK_LOSS = 0.20
LOCALIZATION_SPIN_GAIN = 0.38
TARGET_CONTACT_RADIUS = 0.035
BASE_TARGET_CONTACT_RADIUS = 0.045
LINEAR_ACCEL_LIMIT = 1.10
ANGULAR_ACCEL_LIMIT = 4.80
WHEEL_SPEED_LIMIT = 0.54
WHEEL_ACCEL_LIMIT = 1.35
MIN_TURN_ALIGNMENT = 0.35
MAX_CONTACT_CORRECTION_STEP = 0.022
COSTMAP_SOFT_INFLATION = 0.06
COSTMAP_HARD_MARGIN = 0.018
COSTMAP_MAX_REPULSE_STEP = 0.025
COSTMAP_WARN_INTERVAL_S = 0.80
COSTMAP_LAST_WARN: dict[str, float] = {}
PUSHABLE_OBSTACLE_HALF = OBSTACLE_SIZE * 0.5
PUSHABLE_OBSTACLE_MASS_KG = 1.8
PUSHABLE_OBSTACLE_STATIC_FRICTION = 0.86
PUSHABLE_OBSTACLE_DYNAMIC_FRICTION = 0.68
PUSH_OBSTACLE_STEP_M = 0.060
PUSH_OBSTACLE_CLEARANCE = 0.025
PUSHABLE_OBSTACLE_STARTS = {
    "box_ne": (0.80, 0.80),
    "box_sw": (-0.80, -0.80),
}
OPPONENT_TRACK_RANGE = 3.25
OPPONENT_THREAT_RADIUS = 1.10
OPPONENT_THREAT_BLOCK_THRESHOLD = 0.42
OPPONENT_AVOID_RANGE = 0.38
OPPONENT_AVOID_BEARING_RAD = math.radians(48.0)

YELLOW_ROUTE = [
    YELLOW_START_XY,
    (0.25, -0.78),
    (0.18, -0.22),
    (0.18, 0.20),
    (0.55, 0.20),
    (0.95, 0.20),
    (1.20, 0.22),
]

BLUE_ROUTE = [
    BLUE_START_XY,
    (-0.25, 0.78),
    (-0.18, 0.22),
    (-0.18, -0.20),
    (-0.55, -0.20),
    (-0.95, -0.20),
    (-1.20, -0.22),
]

MATCH_TASKS = {
    "yellow": [
        ("T01_NorthMiddle", (0.42, 1.02)),
        ("T03_WestAboveGate", (-1.20, 0.22)),
        ("T05_EastAboveGate", (1.20, 0.22)),
        ("T02_NorthEast", (1.18, 1.18)),
        ("BlueBaseTarget", YELLOW_ATTACK_BLUE_BASE_XY),
    ],
    "blue": [
        ("T08_SouthMiddle", (-0.42, -1.02)),
        ("T04_WestBelowGate", (-1.20, -0.22)),
        ("T06_EastBelowGate", (1.20, -0.22)),
        ("T07_SouthWest", (-1.18, -1.18)),
        ("YellowBaseTarget", BLUE_ATTACK_YELLOW_BASE_XY),
    ],
}

DEMO_POLICY_TASKS = {
    "yellow": [
        ("T03_WestAboveGate", (-1.16, 0.34)),
        ("T01_NorthMiddle", (0.42, 1.02)),
        ("BlueBaseTarget", YELLOW_ATTACK_BLUE_BASE_XY),
        ("T02_NorthEast", (1.18, 1.05)),
        ("T05_EastAboveGate", (1.16, 0.34)),
    ],
    "blue": [
        ("T06_EastBelowGate", (1.16, -0.34)),
        ("T08_SouthMiddle", (-0.42, -1.02)),
        ("YellowBaseTarget", BLUE_ATTACK_YELLOW_BASE_XY),
        ("T04_WestBelowGate", (-1.16, -0.34)),
        ("T07_SouthWest", (-1.18, -1.05)),
    ],
}

DEMO_FLOW_FIRE_EVENTS: list[tuple[float, str, str]] = [
    (6.25, "yellow", "T05_EastAboveGate"),
    (6.50, "blue", "T04_WestBelowGate"),
    (10.10, "yellow", "T02_NorthEast"),
    (11.05, "blue", "T07_SouthWest"),
    (19.60, "yellow", "T03_WestAboveGate"),
    (21.75, "blue", "T06_EastBelowGate"),
    (25.30, "yellow", "T01_NorthMiddle"),
    (28.45, "blue", "T08_SouthMiddle"),
    (34.80, "yellow", "BlueBaseTarget"),
]

DEMO_FLOW_POSES: dict[str, list[tuple[float, tuple[float, float], str | None]]] = {
    "yellow": [
        (0.00, YELLOW_START_XY, None),
        (2.80, (0.25, -0.62), None),
        (5.35, (0.88, 0.28), "T05_EastAboveGate"),
        (6.65, (0.88, 0.28), "T05_EastAboveGate"),
        (9.20, (1.30, 0.88), "T02_NorthEast"),
        (10.45, (1.30, 0.88), "T02_NorthEast"),
        (13.10, (0.10, 0.07), None),
        (16.70, (0.10, 0.07), None),
        (18.75, (-0.88, 0.28), "T03_WestAboveGate"),
        (20.00, (-0.88, 0.28), "T03_WestAboveGate"),
        (24.50, (0.42, 1.02), "T01_NorthMiddle"),
        (25.70, (0.42, 1.02), "T01_NorthMiddle"),
        (32.55, YELLOW_ATTACK_BLUE_BASE_XY, "BlueBaseTarget"),
        (36.50, YELLOW_ATTACK_BLUE_BASE_XY, "BlueBaseTarget"),
        (42.00, (-0.35, 0.42), "BlueBaseTarget"),
    ],
    "blue": [
        (0.00, BLUE_START_XY, None),
        (2.80, (-0.25, 0.62), None),
        (5.60, (-0.88, -0.28), "T04_WestBelowGate"),
        (7.45, (-0.88, -0.28), "T04_WestBelowGate"),
        (10.20, (-1.30, -0.88), "T07_SouthWest"),
        (11.45, (-1.30, -0.88), "T07_SouthWest"),
        (13.10, (-0.10, -0.07), None),
        (16.70, (-0.10, -0.07), None),
        (20.90, (0.88, -0.28), "T06_EastBelowGate"),
        (22.10, (0.88, -0.28), "T06_EastBelowGate"),
        (27.60, (-0.42, -1.02), "T08_SouthMiddle"),
        (28.80, (-0.42, -1.02), "T08_SouthMiddle"),
        (33.30, BLUE_ATTACK_YELLOW_BASE_XY, "YellowBaseTarget"),
        (42.00, BLUE_ATTACK_YELLOW_BASE_XY, "YellowBaseTarget"),
    ],
}

DEMO_FLOW_RECOVERY_WINDOWS = ((13.00, 16.70),)
DEMO_FLOW_TRIGGERED_EVENTS: set[int] = set()
DEMO_FLOW_PATH_CACHE: dict[tuple[str, int], list[tuple[float, float]]] = {}
TRAINED_REPLAY: dict[str, object] = {
    "loaded": False,
    "rows": {"yellow": [], "blue": []},
    "events": [],
    "applied_events": set(),
    "last_time": 0.0,
    "render_poses": {"yellow": None, "blue": None},
    "last_box_trace_xy": {},
}

