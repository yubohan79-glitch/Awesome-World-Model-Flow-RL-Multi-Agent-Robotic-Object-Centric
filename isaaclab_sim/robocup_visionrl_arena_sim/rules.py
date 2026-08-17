from __future__ import annotations

import math


from ._bootstrap import (
    BASE_ARMOR,
    BASE_HIT_SUCCESS_BY_NORMAL_HITS,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    BLUE_BASE_XY,
    DEMO_POLICY_TASKS,
    LASER_DWELL_FULL_CONFIDENCE_S,
    LASER_DWELL_REQUIRED_S,
    MATCH_STATE,
    MATCH_TASKS,
    NORMAL_SHOOT_MIN_RANGE,
    NORMAL_SHOOT_RANGE,
    OPPONENT_THREAT_RADIUS,
    OPPONENT_TRACK_RANGE,
    TARGET_REGISTRY,
    TARGET_WALL_ANGLE_RAD,
    YELLOW_BASE_XY,
    args_cli
)
from .costmap import wrap_angle
from .laser import line_blocked_by_wall

def opponent_team(team: str) -> str:
    return "blue" if team == "yellow" else "yellow"


def angled_wall_target_yaw(wall_normal_yaw: float, sign: float) -> float:
    return wrap_angle(wall_normal_yaw + sign * TARGET_WALL_ANGLE_RAD)


def target_name_from_path(target_path: str) -> str:
    return target_path.rsplit("/", 1)[-1]


def inward_45deg_target_yaws() -> dict[str, float]:
    # yaw is the target face normal. The visible target plane is yaw + 90 deg,
    # which puts each corner target at 45 deg to both adjacent wall planes.
    return {
        "T01_NorthMiddle": -math.pi / 4.0,
        "T02_NorthEast": -3.0 * math.pi / 4.0,
        "T03_WestAboveGate": math.pi / 4.0,
        "T04_WestBelowGate": -math.pi / 4.0,
        "T05_EastAboveGate": 3.0 * math.pi / 4.0,
        "T06_EastBelowGate": -3.0 * math.pi / 4.0,
        "T07_SouthWest": math.pi / 4.0,
        "T08_SouthMiddle": 3.0 * math.pi / 4.0,
    }


def team_base_xy(team: str) -> tuple[float, float]:
    return YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY


def team_score(team: str) -> int:
    return int(MATCH_STATE[f"score_{team}"])


def normal_hits_against(team: str) -> int:
    opponent = opponent_team(team)
    return max(0, 4 - len(BASE_ARMOR[opponent]))


def base_hit_success_cap(team: str) -> float:
    return float(BASE_HIT_SUCCESS_BY_NORMAL_HITS[max(0, min(4, normal_hits_against(team)))])


def base_removed_side_lane_quality(team: str, target_path: str, fire_xy: tuple[float, float]) -> float:
    target = TARGET_REGISTRY[target_path]
    hits = max(0, min(4, normal_hits_against(team)))
    if hits <= 0:
        return 0.0
    if hits >= 4:
        return 1.0
    base_xy = team_base_xy(str(target["owner"]))
    rel_x = fire_xy[0] - base_xy[0]
    rel_y = fire_xy[1] - base_xy[1]
    distance = math.hypot(rel_x, rel_y)
    if distance < 0.20:
        return 0.0
    unit = (rel_x / max(distance, 1e-6), rel_y / max(distance, 1e-6))
    if base_xy[0] < 0.0:
        opened_dirs = (
            (1.0, 0.0),
            (0.0, -1.0),
            (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
        )
    else:
        opened_dirs = (
            (-1.0, 0.0),
            (0.0, 1.0),
            (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        )
    allowed = opened_dirs[:1] if hits == 1 else opened_dirs[:2] if hits == 2 else opened_dirs
    best_alignment = max(unit[0] * direction[0] + unit[1] * direction[1] for direction in allowed)
    threshold = {1: 0.90, 2: 0.84, 3: 0.58}[hits]
    if best_alignment < threshold:
        return 0.0
    return max(0.0, min(1.0, 0.25 + 0.75 * (best_alignment - threshold) / max(1e-6, 1.0 - threshold)))


def shooting_range_limits(base_target: bool) -> tuple[float, float]:
    if base_target:
        return BASE_SHOOT_MIN_RANGE, BASE_SHOOT_RANGE
    return NORMAL_SHOOT_MIN_RANGE, NORMAL_SHOOT_RANGE


def laser_dwell_success_probability(dwell_s: float) -> float:
    if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
        return 0.0
    alpha = max(
        0.0,
        min(1.0, (dwell_s - LASER_DWELL_REQUIRED_S) / (LASER_DWELL_FULL_CONFIDENCE_S - LASER_DWELL_REQUIRED_S)),
    )
    not_fall_probability = 0.20 - 0.10 * alpha
    return max(0.0, min(0.90, 1.0 - not_fall_probability))


def normalized_laser_dwell_factor(dwell_s: float) -> float:
    return laser_dwell_success_probability(dwell_s) / 0.90


def base_attack_pose_quality(team: str, target_path: str, fire_xy: tuple[float, float]) -> float:
    target = TARGET_REGISTRY[target_path]
    if not str(target["kind"]).startswith("base_"):
        return 1.0
    hits = normal_hits_against(team)
    if hits <= 0:
        return 0.0
    side_quality = base_removed_side_lane_quality(team, target_path, fire_xy)
    if side_quality <= 0.0:
        return 0.0
    x, y = fire_xy
    target_xy = target["xy"]
    assert isinstance(target_xy, tuple)
    approach_yaw = math.atan2(y - target_xy[1], x - target_xy[0])
    off_axis = abs(wrap_angle(approach_yaw - float(target["yaw"])))
    min_off_axis = {1: 0.62, 2: 0.42, 3: 0.18, 4: 0.0}[max(1, min(4, hits))]
    max_off_axis = 2.55
    if off_axis < min_off_axis or off_axis > max_off_axis:
        return 0.0
    base_xy = team_base_xy(str(target["owner"]))
    base_distance = math.hypot(base_xy[0] - x, base_xy[1] - y)
    corner_radius = {1: 0.95, 2: 1.05, 3: 1.22, 4: 1.45}[max(1, min(4, hits))]
    if base_distance > corner_radius:
        return 0.0
    angle_quality = (off_axis - min_off_axis) / max(max_off_axis - min_off_axis, 1e-6)
    corner_quality = 1.0 - base_distance / max(corner_radius, 1e-6)
    # Early base rushes are allowed, but only through narrow off-axis lanes
    # around the grounded armor. More removed armor widens the acceptable angle.
    return max(0.0, min(1.0, (0.38 + 0.37 * angle_quality + 0.25 * corner_quality) * side_quality))


def static_fire_pose(team: str, target_name: str, tasks: list[tuple[str, tuple[float, float]]] | None = None) -> tuple[float, float] | None:
    task_table = tasks if tasks is not None else (DEMO_POLICY_TASKS[team] if args_cli.demo_flow else MATCH_TASKS[team])
    for candidate_name, fire_xy in task_table:
        if candidate_name == target_name:
            return fire_xy
    return None


def empty_opponent_estimate() -> dict[str, float | bool]:
    return {
        "available": False,
        "visible": False,
        "dx": 0.0,
        "dy": 0.0,
        "distance": OPPONENT_TRACK_RANGE,
        "global_bearing": 0.0,
        "relative_bearing": 0.0,
        "relative_heading": 0.0,
        "distance_to_own_base": OPPONENT_THREAT_RADIUS,
        "heading_to_own_base": 0.0,
        "threat_to_own_base": 0.0,
    }


def opponent_bearing_estimate(
    team: str,
    own_pose: tuple[tuple[float, float, float], float],
    opponent_pose: tuple[tuple[float, float, float], float],
) -> dict[str, float | bool]:
    own_pos, own_yaw = own_pose
    opponent_pos, opponent_yaw = opponent_pose
    dx = opponent_pos[0] - own_pos[0]
    dy = opponent_pos[1] - own_pos[1]
    distance = math.hypot(dx, dy)
    global_bearing = math.atan2(dy, dx) if distance > 1e-6 else own_yaw
    relative_bearing = wrap_angle(global_bearing - own_yaw)
    relative_heading = wrap_angle(opponent_yaw - own_yaw)
    line_of_sight = not line_blocked_by_wall((own_pos[0], own_pos[1]), (opponent_pos[0], opponent_pos[1]))
    visible = distance <= OPPONENT_TRACK_RANGE and line_of_sight

    own_base = team_base_xy(team)
    base_dx = own_base[0] - opponent_pos[0]
    base_dy = own_base[1] - opponent_pos[1]
    distance_to_own_base = math.hypot(base_dx, base_dy)
    base_bearing_from_opponent = math.atan2(base_dy, base_dx) if distance_to_own_base > 1e-6 else opponent_yaw
    heading_to_own_base = abs(wrap_angle(base_bearing_from_opponent - opponent_yaw))
    proximity_threat = max(0.0, 1.0 - distance_to_own_base / OPPONENT_THREAT_RADIUS)
    heading_threat = max(0.0, 1.0 - heading_to_own_base / math.pi)
    visibility_scale = 1.0 if visible else 0.72
    threat_to_own_base = max(0.0, min(1.0, proximity_threat * (0.55 + 0.45 * heading_threat) * visibility_scale))

    return {
        "available": True,
        "visible": visible,
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "global_bearing": global_bearing,
        "relative_bearing": relative_bearing,
        "relative_heading": relative_heading,
        "distance_to_own_base": distance_to_own_base,
        "heading_to_own_base": heading_to_own_base,
        "threat_to_own_base": threat_to_own_base,
    }
