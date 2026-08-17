from __future__ import annotations

import math

import numpy as np

from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BASE_ARMOR_SPECS,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    active_base_armor_blockers,
)
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


MAX_TARGETS = 10
TARGET_FEATURE_DIM = 9
MAX_PUSHABLE_BOXES = 2
BOX_FEATURE_DIM = 5
MAX_ARMOR_BLOCKERS = 8
BLOCKER_FEATURE_DIM = 5
ROBOT_FEATURE_DIM = 8
GLOBAL_FEATURE_DIM = 9

OBJECT_STATE_DIM = (
    GLOBAL_FEATURE_DIM
    + len(AGENTS) * ROBOT_FEATURE_DIM
    + MAX_TARGETS * TARGET_FEATURE_DIM
    + MAX_PUSHABLE_BOXES * BOX_FEATURE_DIM
    + MAX_ARMOR_BLOCKERS * BLOCKER_FEATURE_DIM
)


def _team_id(team: str) -> float:
    return 1.0 if team == "yellow" else -1.0


def _target_kind_id(kind: str) -> float:
    if kind == "normal":
        return 0.0
    if kind == "base_yellow":
        return -1.0
    if kind == "base_blue":
        return 1.0
    return 0.0


def _target_sort_key(target) -> tuple[int, str]:
    owner_index = 0 if target.owner == "yellow" else 1
    kind_index = 0 if target.kind == "normal" else 1
    return owner_index * 20 + kind_index, str(target.name)


def _pad(values: list[np.ndarray], count: int, dim: int) -> np.ndarray:
    rows = values[:count]
    while len(rows) < count:
        rows.append(np.zeros(dim, dtype=np.float32))
    return np.concatenate(rows).astype(np.float32) if rows else np.zeros(count * dim, dtype=np.float32)


def _robot_features(env: RoboCupVisionRLSelfPlayEnv, team: str) -> np.ndarray:
    pose = env.poses[team]
    opponent = "blue" if team == "yellow" else "yellow"
    normal_hits = max(0, 4 - int(env.armor[opponent]))
    fusion = env.sensor_fusion.get(team, {})
    return np.array(
        [
            float(pose[0]) / HALF_ARENA,
            float(pose[1]) / HALF_ARENA,
            math.cos(float(pose[2])),
            math.sin(float(pose[2])),
            float(env.armor[team]) / 4.0,
            float(normal_hits) / 4.0,
            float(env.localization_confidence.get(team, 1.0)),
            float(fusion.get("pushable_contact", 0.0)),
        ],
        dtype=np.float32,
    )


def _target_features(env: RoboCupVisionRLSelfPlayEnv) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for target in sorted(env.targets, key=_target_sort_key):
        xy = np.asarray(target.xy, dtype=np.float32)
        owner = _team_id(target.owner)
        rows.append(
            np.array(
                [
                    float(xy[0]) / HALF_ARENA,
                    float(xy[1]) / HALF_ARENA,
                    math.cos(float(target.yaw)),
                    math.sin(float(target.yaw)),
                    owner,
                    _target_kind_id(target.kind),
                    1.0 if bool(target.knocked) else 0.0,
                    1.0 if env._line_blocked((float(env.poses["yellow"][0]), float(env.poses["yellow"][1])), target.xy) else 0.0,
                    1.0 if env._line_blocked((float(env.poses["blue"][0]), float(env.poses["blue"][1])), target.xy) else 0.0,
                ],
                dtype=np.float32,
            )
        )
    return rows


def _box_features(env: RoboCupVisionRLSelfPlayEnv) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    starts = getattr(env, "pushable_obstacles", {})
    for name in sorted(starts):
        xy = np.asarray(starts[name], dtype=np.float32)
        rows.append(
            np.array(
                [
                    float(xy[0]) / HALF_ARENA,
                    float(xy[1]) / HALF_ARENA,
                    PUSHABLE_OBSTACLE_HALF / HALF_ARENA,
                    PUSHABLE_OBSTACLE_HALF / HALF_ARENA,
                    1.0,
                ],
                dtype=np.float32,
            )
        )
    return rows


def _blocker_features(env: RoboCupVisionRLSelfPlayEnv) -> list[np.ndarray]:
    active = set()
    for center, half in active_base_armor_blockers(env.armor, inflated=False):
        active.add((round(float(center[0]), 4), round(float(center[1]), 4), round(float(half[0]), 4), round(float(half[1]), 4)))

    rows: list[np.ndarray] = []
    for team in ("yellow", "blue"):
        for center, size in BASE_ARMOR_SPECS[team]:
            half = (float(size[0]) * 0.5, float(size[1]) * 0.5)
            key = (round(float(center[0]), 4), round(float(center[1]), 4), round(float(half[0]), 4), round(float(half[1]), 4))
            rows.append(
                np.array(
                    [
                        float(center[0]) / HALF_ARENA,
                        float(center[1]) / HALF_ARENA,
                        half[0] / HALF_ARENA,
                        half[1] / HALF_ARENA,
                        1.0 if key in active else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
    return rows


def extract_object_state(env: RoboCupVisionRLSelfPlayEnv) -> np.ndarray:
    """Return a fixed object-centric state vector for world-model and critic use.

    The local actor can remain decentralized, but critics/world models receive
    explicit robot, target, box and armor-blocker tokens instead of only the
    flattened local observations.
    """

    score_delta = float(env.scores["yellow"] - env.scores["blue"]) / 60.0
    robot_distance = float(np.linalg.norm(env.poses["yellow"][:2] - env.poses["blue"][:2])) / ARENA_SIZE
    global_features = np.array(
        [
            float(env.elapsed) / max(float(env.max_time_s), 1e-6),
            score_delta,
            float(env.scores["yellow"]) / 60.0,
            float(env.scores["blue"]) / 60.0,
            float(env.armor["yellow"]) / 4.0,
            float(env.armor["blue"]) / 4.0,
            robot_distance,
            1.0 if bool(env.last_contact) else 0.0,
            1.0 if env.winner is not None else 0.0,
        ],
        dtype=np.float32,
    )
    pieces = [
        global_features,
        np.concatenate([_robot_features(env, team) for team in AGENTS]).astype(np.float32),
        _pad(_target_features(env), MAX_TARGETS, TARGET_FEATURE_DIM),
        _pad(_box_features(env), MAX_PUSHABLE_BOXES, BOX_FEATURE_DIM),
        _pad(_blocker_features(env), MAX_ARMOR_BLOCKERS, BLOCKER_FEATURE_DIM),
    ]
    state = np.concatenate(pieces).astype(np.float32)
    if state.shape[0] != OBJECT_STATE_DIM:
        raise RuntimeError(f"object state dim mismatch: {state.shape[0]} != {OBJECT_STATE_DIM}")
    return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
