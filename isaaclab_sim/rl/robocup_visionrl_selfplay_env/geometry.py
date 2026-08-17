from __future__ import annotations

import math

import numpy as np

from .constants import (
    ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS
)
from robocup_visionrl_gym_env import (
    PUSHABLE_OBSTACLE_HALF,
    SHOOTER_FORWARD_OFFSET
)

def laser_origin_from_pose(pose: np.ndarray) -> tuple[float, float]:
    yaw = float(pose[2])
    return (
        float(pose[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
        float(pose[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
    )


def circle_aabb_collision(
    point: tuple[float, float],
    center: tuple[float, float],
    half_size: tuple[float, float],
    radius: float,
) -> tuple[bool, tuple[float, float], float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    closest_x = max(center[0] - half_size[0], min(point[0], center[0] + half_size[0]))
    closest_y = max(center[1] - half_size[1], min(point[1], center[1] + half_size[1]))
    vx = point[0] - closest_x
    vy = point[1] - closest_y
    distance = math.hypot(vx, vy)
    if distance > 1e-8:
        penetration = radius - distance
        if penetration <= 0.0:
            return False, (0.0, 0.0), 0.0
        return True, (vx / distance, vy / distance), penetration

    inside_x = half_size[0] - abs(dx)
    inside_y = half_size[1] - abs(dy)
    if inside_x < 0.0 or inside_y < 0.0:
        return False, (0.0, 0.0), 0.0
    if inside_x <= inside_y:
        normal = (1.0 if dx >= 0.0 else -1.0, 0.0)
        penetration = inside_x + radius
    else:
        normal = (0.0, 1.0 if dy >= 0.0 else -1.0)
        penetration = inside_y + radius
    return True, normal, penetration


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
    return oriented_rect_aabb_collision(
        (float(pose[0]), float(pose[1])),
        float(pose[2]),
        ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
        box_center,
        box_half,
    )


def team_frame_sign(team: str) -> float:
    return 1.0 if team == "yellow" else -1.0
