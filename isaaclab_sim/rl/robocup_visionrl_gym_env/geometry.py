from __future__ import annotations

import math

import numpy as np

from .constants import (
    BASE_ARMOR_SPECS,
    BASE_HIT_RADIUS,
    BASE_HIT_SUCCESS_BY_NORMAL_HITS,
    BASE_SHOOT_IDEAL_DISTANCE,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    LASER_DWELL_FULL_CONFIDENCE_S,
    LASER_DWELL_REQUIRED_S,
    NORMAL_SHOOT_IDEAL_DISTANCE,
    NORMAL_SHOOT_MIN_RANGE,
    NORMAL_SHOOT_RANGE,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
    ROUTE_CLEARANCE,
    SHOOTER_FORWARD_OFFSET,
    SHOOT_HIT_RADIUS,
    TARGET_WALL_ANGLE_RAD
)

def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def oriented_rect_aabb_collision(
    rect_center: tuple[float, float],
    yaw: float,
    rect_half: tuple[float, float],
    box_center: tuple[float, float],
    box_half: tuple[float, float],
) -> tuple[bool, tuple[float, float], float]:
    ux = (math.cos(yaw), math.sin(yaw))
    uy = (-math.sin(yaw), math.cos(yaw))
    delta = (rect_center[0] - box_center[0], rect_center[1] - box_center[1])
    axes = (ux, uy, (1.0, 0.0), (0.0, 1.0))
    best_axis = (1.0, 0.0)
    best_overlap = math.inf
    for axis in axes:
        rect_radius = rect_half[0] * abs(ux[0] * axis[0] + ux[1] * axis[1]) + rect_half[1] * abs(
            uy[0] * axis[0] + uy[1] * axis[1]
        )
        box_radius = box_half[0] * abs(axis[0]) + box_half[1] * abs(axis[1])
        distance = abs(delta[0] * axis[0] + delta[1] * axis[1])
        overlap = rect_radius + box_radius - distance
        if overlap <= 0.0:
            return False, (0.0, 0.0), 0.0
        if overlap < best_overlap:
            best_overlap = overlap
            sign = 1.0 if delta[0] * axis[0] + delta[1] * axis[1] >= 0.0 else -1.0
            best_axis = (axis[0] * sign, axis[1] * sign)
    norm = math.hypot(best_axis[0], best_axis[1])
    if norm <= 1e-8:
        return True, (1.0, 0.0), float(best_overlap)
    return True, (best_axis[0] / norm, best_axis[1] / norm), float(best_overlap)


def robot_pushable_collision(
    pose: np.ndarray,
    box_center: tuple[float, float],
    box_half: tuple[float, float] = (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
) -> tuple[bool, tuple[float, float], float]:
    yaw = float(pose[2]) if pose.shape[0] >= 3 else 0.0
    return oriented_rect_aabb_collision(
        (float(pose[0]), float(pose[1])),
        yaw,
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        box_center,
        box_half,
    )


def angled_wall_target_yaw(wall_normal_yaw: float, sign: float) -> float:
    return wrap_angle(wall_normal_yaw + sign * TARGET_WALL_ANGLE_RAD)


def inward_45deg_target_yaws() -> dict[str, float]:
    # yaw is the target face normal. The target plane itself is yaw + 90 deg,
    # so each corner panel cuts the two wall planes at 45 deg.
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


def route_pose(t: float, route: list[tuple[float, float]], speed: float = 0.22) -> np.ndarray:
    segment_lengths = [
        math.hypot(route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1])
        for i in range(len(route) - 1)
    ]
    total_length = sum(segment_lengths)
    travel = (t * speed) % (total_length * 2.0)
    reverse = travel > total_length
    distance = total_length * 2.0 - travel if reverse else travel
    walked = 0.0
    for index, length in enumerate(segment_lengths):
        if distance <= walked + length or index == len(segment_lengths) - 1:
            alpha = 0.0 if length <= 1e-9 else (distance - walked) / length
            eased = 0.5 - 0.5 * math.cos(max(0.0, min(1.0, alpha)) * math.pi)
            x0, y0 = route[index]
            x1, y1 = route[index + 1]
            yaw = math.atan2(y1 - y0, x1 - x0)
            if reverse:
                yaw += math.pi
            return np.array([x0 + (x1 - x0) * eased, y0 + (y1 - y0) * eased, wrap_angle(yaw)], dtype=np.float32)
        walked += length
    return np.array([route[-1][0], route[-1][1], 0.0], dtype=np.float32)


def laser_origin_from_pose(pose: np.ndarray) -> tuple[float, float]:
    yaw = float(pose[2])
    return (
        float(pose[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
        float(pose[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
    )


def shooting_range_limits(base_target: bool) -> tuple[float, float]:
    if base_target:
        return BASE_SHOOT_MIN_RANGE, BASE_SHOOT_RANGE
    return NORMAL_SHOOT_MIN_RANGE, NORMAL_SHOOT_RANGE


def ideal_shoot_distance(base_target: bool) -> float:
    return BASE_SHOOT_IDEAL_DISTANCE if base_target else NORMAL_SHOOT_IDEAL_DISTANCE


def laser_accuracy_from_geometry(distance: float, lateral_error: float, base_target: bool) -> float:
    min_range, max_range = shooting_range_limits(base_target)
    if distance < min_range or distance > max_range:
        return 0.0
    hit_radius = BASE_HIT_RADIUS if base_target else SHOOT_HIT_RADIUS
    if lateral_error > hit_radius:
        return 0.0
    distance_quality = (max_range - distance) / max(1e-6, max_range - min_range)
    lateral_quality = 1.0 - lateral_error / max(hit_radius, 1e-6)
    accuracy = 0.18 + 0.64 * distance_quality + 0.18 * lateral_quality
    if base_target:
        accuracy -= 0.10
    return float(np.clip(accuracy, 0.05, 0.98))


def laser_dwell_success_probability(dwell_s: float) -> float:
    if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
        return 0.0
    alpha = min(1.0, max(0.0, (dwell_s - LASER_DWELL_REQUIRED_S) / (LASER_DWELL_FULL_CONFIDENCE_S - LASER_DWELL_REQUIRED_S)))
    not_fall = 0.20 - 0.10 * alpha
    return float(np.clip(1.0 - not_fall, 0.0, 0.90))


def normalized_laser_dwell_factor(dwell_s: float) -> float:
    return laser_dwell_success_probability(dwell_s) / 0.90


def base_hit_success_cap(normal_hits: int) -> float:
    key = max(0, min(4, int(normal_hits)))
    return float(BASE_HIT_SUCCESS_BY_NORMAL_HITS[key])


def base_removed_side_lane_quality(normal_hits: int, base_xy: np.ndarray, xy: np.ndarray) -> float:
    """Score whether a base shot is taken from the side whose armor was removed.

    The four armor plates open the base progressively. A one-target early rush
    may only shoot through the first removed side; after two normal hits the
    second side is also allowed. This prevents far or arbitrary line-of-sight
    shots from counting as a legal base attack.
    """

    hits = max(0, min(4, int(normal_hits)))
    if hits <= 0:
        return 0.0
    if hits >= 4:
        return 1.0
    base = np.asarray(base_xy, dtype=np.float32)
    point = np.asarray(xy, dtype=np.float32)
    rel = point - base
    distance = float(np.linalg.norm(rel))
    if distance < 0.20:
        return 0.0
    unit = rel / max(distance, 1e-6)
    if float(base[0]) < 0.0:
        opened_dirs = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([1.0, -1.0], dtype=np.float32) / math.sqrt(2.0),
        ]
    else:
        opened_dirs = [
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 1.0], dtype=np.float32) / math.sqrt(2.0),
        ]
    allowed = opened_dirs[:1] if hits == 1 else opened_dirs[:2] if hits == 2 else opened_dirs
    best_alignment = max(float(np.dot(unit, direction)) for direction in allowed)
    threshold = {1: 0.90, 2: 0.84, 3: 0.58}[hits]
    if best_alignment < threshold:
        return 0.0
    return float(np.clip(0.25 + 0.75 * (best_alignment - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0))


def base_attack_pose_quality(normal_hits: int, target_xy: tuple[float, float], target_yaw: float, base_xy: np.ndarray, xy: np.ndarray) -> float:
    hits = max(0, min(4, int(normal_hits)))
    if hits <= 0:
        return 0.0
    side_quality = base_removed_side_lane_quality(hits, base_xy, xy)
    if side_quality <= 0.0:
        return 0.0
    approach_yaw = math.atan2(float(xy[1]) - target_xy[1], float(xy[0]) - target_xy[0])
    off_axis = abs(wrap_angle(approach_yaw - float(target_yaw)))
    min_off_axis = {1: 0.62, 2: 0.42, 3: 0.18, 4: 0.0}[hits]
    max_off_axis = 2.55
    if off_axis < min_off_axis or off_axis > max_off_axis:
        return 0.0
    base_distance = float(np.linalg.norm(np.asarray(base_xy, dtype=np.float32) - np.asarray(xy, dtype=np.float32)))
    corner_radius = {1: 0.95, 2: 1.05, 3: 1.22, 4: 1.45}[hits]
    if base_distance > corner_radius:
        return 0.0
    angle_quality = (off_axis - min_off_axis) / max(max_off_axis - min_off_axis, 1e-6)
    corner_quality = 1.0 - base_distance / max(corner_radius, 1e-6)
    return float(np.clip((0.38 + 0.37 * angle_quality + 0.25 * corner_quality) * side_quality, 0.0, 1.0))


def active_base_armor_blockers(
    armor_remaining: dict[str, int],
    *,
    inflated: bool = False,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    margin = ROUTE_CLEARANCE + 0.045 if inflated else 0.0
    blockers: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for team, specs in BASE_ARMOR_SPECS.items():
        remaining = max(0, min(4, int(armor_remaining.get(team, 4))))
        for center, size in specs[4 - remaining :]:
            blockers.append((center, (size[0] * 0.5 + margin, size[1] * 0.5 + margin)))
    return blockers
