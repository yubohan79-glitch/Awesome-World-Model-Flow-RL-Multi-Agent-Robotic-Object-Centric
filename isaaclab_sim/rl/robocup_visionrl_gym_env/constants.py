from __future__ import annotations

import math

import numpy as np

ARENA_SIZE = 3.0
HALF_ARENA = ARENA_SIZE * 0.5
WALL_THICKNESS = 0.04
ZONE_SIZE = 0.50
OBSTACLE_SIZE = 0.30
PUSHABLE_OBSTACLE_HALF = OBSTACLE_SIZE * 0.5
# The rules state that two 30 cm cube obstacles are randomly placed.  The
# default deterministic layout follows the red obstacle centers measured from
# the national-rule field diagram; training can then jitter these references.
PUSHABLE_OBSTACLE_STARTS = {
    "box_ne": np.array([0.80, 0.80], dtype=np.float32),
    "box_sw": np.array([-0.80, -0.80], dtype=np.float32),
}
PUSHABLE_OBSTACLE_RANDOM_JITTER = 0.08
PUSHABLE_STEP_M = 0.060
PUSHABLE_CLEARANCE_MARGIN = 0.025
ROBOT_LENGTH = 0.34
ROBOT_WIDTH = 0.24
ROBOT_RADIUS = math.hypot(ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5)
ROBOT_PUSHABLE_CLEARANCE_RADIUS = ROBOT_RADIUS + 0.030
# Conservative visual/contact hull used for pushable red boxes.  It includes
# the rendered wheel/body footprint so videos and strict audits agree.
ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS = (ROBOT_LENGTH * 0.5 + 0.110, ROBOT_WIDTH * 0.5 + 0.087)
ROUTE_CLEARANCE = ROBOT_WIDTH * 0.5 + 0.04
# Real-laser contract used by the RL rule environments and the IsaacLab replay.
# Distances are measured from the fixed shooter outlet, not from base_link.
# Normal targets remain a close 5-50 cm shot. Base targets are physically
# recessed behind armor, so the valid outlet-to-target range is wider but still
# bounded; line-of-sight through remaining armor is always checked separately.
NORMAL_SHOOT_MIN_RANGE = 0.05
NORMAL_SHOOT_RANGE = 0.50
NORMAL_SHOOT_IDEAL_DISTANCE = 0.30
BASE_SHOOT_MIN_RANGE = 0.20
BASE_SHOOT_RANGE = 0.80
BASE_SHOOT_IDEAL_DISTANCE = 0.48
SHOOT_MIN_RANGE = NORMAL_SHOOT_MIN_RANGE
SHOOT_RANGE = NORMAL_SHOOT_RANGE
SHOOT_IDEAL_DISTANCE = NORMAL_SHOOT_IDEAL_DISTANCE
SHOOTER_FORWARD_OFFSET = 0.20
SHOOT_HIT_RADIUS = 0.028
BASE_HIT_RADIUS = 0.018
NORMAL_TARGET_CONTACT_RADIUS = 0.035
BASE_TARGET_CONTACT_RADIUS = 0.045
LASER_DWELL_REQUIRED_S = 0.80
LASER_DWELL_FULL_CONFIDENCE_S = 2.00
LASER_FIRE_COOLDOWN_S = 1.0
TARGET_WALL_INSET = 0.240
TARGET_WALL_ANGLE_RAD = math.radians(45.0)
NORTH_MIDDLE_TARGET_X = 0.18
SOUTH_MIDDLE_TARGET_X = -0.18
SIDE_GATE_TARGET_Y = 0.24

YELLOW_START = np.array([0.25, -1.25, math.pi * 0.5], dtype=np.float32)
BLUE_START = np.array([-0.25, 1.25, -math.pi * 0.5], dtype=np.float32)
BLUE_BASE_XY = np.array([-1.25, 1.25], dtype=np.float32)
YELLOW_BASE_XY = np.array([1.25, -1.25], dtype=np.float32)
BLUE_BASE_TARGET_XY = np.array([-1.36, 1.36], dtype=np.float32)
YELLOW_BASE_TARGET_XY = np.array([1.36, -1.36], dtype=np.float32)
BLUE_BASE_TARGET_YAW = -math.pi / 4.0
YELLOW_BASE_TARGET_YAW = 3.0 * math.pi / 4.0
BASE_HIT_SUCCESS_BY_NORMAL_HITS = {
    0: 0.0,
    1: 0.40,
    2: 0.55,
    3: 0.80,
    4: 0.95,
}

BASE_ARMOR_SIZE = {
    "thickness": 0.050,
    "length": 0.250,
}
BASE_ARMOR_SPECS = {
    "blue": [
        ((-1.025, 1.375), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((-1.375, 1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
        ((-1.025, 1.125), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((-1.125, 1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
    ],
    "yellow": [
        ((1.025, -1.375), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((1.375, -1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
        ((1.025, -1.125), (BASE_ARMOR_SIZE["thickness"], BASE_ARMOR_SIZE["length"])),
        ((1.125, -1.025), (BASE_ARMOR_SIZE["length"], BASE_ARMOR_SIZE["thickness"])),
    ],
}

BLUE_ROUTE = [
    (-0.25, 1.25),
    (-0.25, 0.78),
    (-0.18, 0.22),
    (-0.18, -0.20),
    (-0.55, -0.20),
    (-0.95, -0.20),
    (-1.20, -0.22),
]
