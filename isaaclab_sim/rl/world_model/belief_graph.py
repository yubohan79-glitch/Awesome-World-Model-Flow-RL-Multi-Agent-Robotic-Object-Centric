from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BASE_ARMOR_SPECS,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    active_base_armor_blockers,
    segment_intersects_aabb,
)
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv

from .object_state import (
    MAX_ARMOR_BLOCKERS,
    MAX_PUSHABLE_BOXES,
    MAX_TARGETS,
    _target_sort_key,
)


class NodeType(IntEnum):
    GLOBAL = 0
    ROBOT = 1
    TARGET = 2
    BOX = 3
    ARMOR_BLOCKER = 4


class EdgeType(IntEnum):
    GLOBAL = 0
    OBSERVES = 1
    CONTACTS = 2
    BLOCKS_ROUTE = 3
    PROTECTS_BASE = 4
    THREATENS = 5
    PROXIMITY = 6
    LINE_OF_SIGHT = 7


class RuleRisk(IntEnum):
    ROBOT_OR_TARGET_COLLISION = 0
    BLOCKED_OR_PENETRATION = 1
    ILLEGAL_OR_OWN_TARGET_FIRE = 2
    LOS_OR_RANGE_VIOLATION = 3


NUM_NODE_TYPES = len(NodeType)
NUM_EDGE_TYPES = len(EdgeType)
NUM_RULE_RISKS = len(RuleRisk)

TOKEN_X = 0
TOKEN_Y = 1
TOKEN_COS_YAW = 2
TOKEN_SIN_YAW = 3
TOKEN_VX = 4
TOKEN_VY = 5
TOKEN_ATTRIBUTE_A = 6
TOKEN_ATTRIBUTE_B = 7
TOKEN_EXTENT_X = 8
TOKEN_EXTENT_Y = 9
TOKEN_VISIBLE = 10
TOKEN_LAST_SEEN = 11
TOKEN_AGE = 12
TOKEN_COVARIANCE = 13
TOKEN_OCCLUDED = 14
TOKEN_PRESENT = 15
BELIEF_TOKEN_DIM = 16
PHYSICAL_TOKEN_DIM = 10

GLOBAL_NODE_COUNT = 1
ROBOT_NODE_COUNT = len(AGENTS)
NUM_BELIEF_NODES = (
    GLOBAL_NODE_COUNT
    + ROBOT_NODE_COUNT
    + MAX_TARGETS
    + MAX_PUSHABLE_BOXES
    + MAX_ARMOR_BLOCKERS
)
BELIEF_STATE_DIM = NUM_BELIEF_NODES * BELIEF_TOKEN_DIM

GLOBAL_SLICE = slice(0, 1)
ROBOT_SLICE = slice(GLOBAL_SLICE.stop, GLOBAL_SLICE.stop + ROBOT_NODE_COUNT)
TARGET_SLICE = slice(ROBOT_SLICE.stop, ROBOT_SLICE.stop + MAX_TARGETS)
BOX_SLICE = slice(TARGET_SLICE.stop, TARGET_SLICE.stop + MAX_PUSHABLE_BOXES)
ARMOR_BLOCKER_SLICE = slice(BOX_SLICE.stop, BOX_SLICE.stop + MAX_ARMOR_BLOCKERS)


def canonical_node_types() -> np.ndarray:
    values = (
        [NodeType.GLOBAL]
        + [NodeType.ROBOT] * ROBOT_NODE_COUNT
        + [NodeType.TARGET] * MAX_TARGETS
        + [NodeType.BOX] * MAX_PUSHABLE_BOXES
        + [NodeType.ARMOR_BLOCKER] * MAX_ARMOR_BLOCKERS
    )
    return np.asarray(values, dtype=np.int64)


@dataclass
class BeliefGraph:
    tokens: np.ndarray
    node_types: np.ndarray

    def flatten(self) -> np.ndarray:
        return self.tokens.reshape(-1).astype(np.float32, copy=False)


@dataclass
class _Measurement:
    features: np.ndarray
    present: np.ndarray
    visible: np.ndarray
    occluded: np.ndarray
    timestamp: float


def _team_id(team: str) -> float:
    return 1.0 if team == "yellow" else -1.0


def _target_kind_id(kind: str) -> float:
    if kind == "base_yellow":
        return -1.0
    if kind == "base_blue":
        return 1.0
    return 0.0


def _line_occluded(
    env: RoboCupVisionRLSelfPlayEnv,
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    ignored_box: str | None = None,
) -> bool:
    for center, half_size in env.laser_blockers:
        if segment_intersects_aabb(origin, destination, center, half_size):
            return True
    for center, half_size in active_base_armor_blockers(env.armor, inflated=False):
        if segment_intersects_aabb(origin, destination, center, half_size):
            return True
    for name, center in env.pushable_obstacles.items():
        if name == ignored_box:
            continue
        if segment_intersects_aabb(
            origin,
            destination,
            (float(center[0]), float(center[1])),
            (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
        ):
            return True
    return False


def _visible_from_any_robot(
    env: RoboCupVisionRLSelfPlayEnv,
    xy: tuple[float, float],
    *,
    ignored_box: str | None = None,
) -> bool:
    for team in AGENTS:
        pose = env.poses[team]
        origin = (float(pose[0]), float(pose[1]))
        if np.linalg.norm(np.asarray(xy, dtype=np.float32) - pose[:2]) > ARENA_SIZE:
            continue
        if not _line_occluded(env, origin, xy, ignored_box=ignored_box):
            return True
    return False


class BeliefTracker:
    """Maintains sensor-facing object beliefs without exposing truth when occluded.

    The simulator supplies measurements through a small adapter. A ROS adapter can
    populate the same token contract from detections, timestamps and covariances.
    Privileged simulator state is only used to form a measurement when the object is
    visible; otherwise the tracker propagates its previous belief and uncertainty.
    """

    def __init__(
        self,
        *,
        max_age_s: float = 3.0,
        covariance_growth: float = 0.08,
        sensor_delay_steps: int = 0,
        observation_dropout: float = 0.0,
        uncertainty_enabled: bool = True,
        seed: int = 0,
    ):
        self.max_age_s = max(float(max_age_s), 1e-3)
        self.covariance_growth = max(float(covariance_growth), 0.0)
        self.sensor_delay_steps = max(int(sensor_delay_steps), 0)
        self.observation_dropout = float(np.clip(observation_dropout, 0.0, 1.0))
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.rng = np.random.default_rng(seed)
        self.node_types = canonical_node_types()
        self._history: list[_Measurement] = []
        self.reset()

    def reset(self) -> None:
        self.tokens = np.zeros((NUM_BELIEF_NODES, BELIEF_TOKEN_DIM), dtype=np.float32)
        self.last_positions = np.zeros((NUM_BELIEF_NODES, 2), dtype=np.float32)
        self.last_measurement_time = np.full(NUM_BELIEF_NODES, -np.inf, dtype=np.float32)
        self._history = []

    def _measurement(self, env: RoboCupVisionRLSelfPlayEnv) -> _Measurement:
        features = np.zeros((NUM_BELIEF_NODES, PHYSICAL_TOKEN_DIM), dtype=np.float32)
        present = np.zeros(NUM_BELIEF_NODES, dtype=np.float32)
        visible = np.zeros(NUM_BELIEF_NODES, dtype=np.float32)
        occluded = np.zeros(NUM_BELIEF_NODES, dtype=np.float32)
        elapsed_norm = float(env.elapsed) / max(float(env.max_time_s), 1e-6)

        features[0] = np.asarray(
            [
                elapsed_norm,
                float(env.scores["yellow"] - env.scores["blue"]) / 60.0,
                float(env.scores["yellow"]) / 60.0,
                float(env.scores["blue"]) / 60.0,
                0.0,
                0.0,
                float(env.armor["yellow"]) / 4.0,
                float(env.armor["blue"]) / 4.0,
                float(env.last_contact),
                float(env.winner is not None),
            ],
            dtype=np.float32,
        )
        present[0] = visible[0] = 1.0

        for offset, team in enumerate(AGENTS):
            index = ROBOT_SLICE.start + offset
            pose = env.poses[team]
            confidence = float(env.localization_confidence.get(team, 1.0))
            features[index] = np.asarray(
                [
                    float(pose[0]) / HALF_ARENA,
                    float(pose[1]) / HALF_ARENA,
                    math.cos(float(pose[2])),
                    math.sin(float(pose[2])),
                    0.0,
                    0.0,
                    _team_id(team),
                    float(env.armor[team]) / 4.0,
                    0.10 / HALF_ARENA,
                    0.10 / HALF_ARENA,
                ],
                dtype=np.float32,
            )
            present[index] = 1.0
            visible[index] = 1.0 if confidence >= 0.15 else 0.0
            occluded[index] = 1.0 - visible[index]

        targets = sorted(env.targets, key=_target_sort_key)[:MAX_TARGETS]
        for offset, target in enumerate(targets):
            index = TARGET_SLICE.start + offset
            xy = (float(target.xy[0]), float(target.xy[1]))
            target_present = not bool(target.knocked)
            target_visible = _visible_from_any_robot(env, xy)
            features[index] = np.asarray(
                [
                    xy[0] / HALF_ARENA,
                    xy[1] / HALF_ARENA,
                    math.cos(float(target.yaw)),
                    math.sin(float(target.yaw)),
                    0.0,
                    0.0,
                    _team_id(target.owner),
                    _target_kind_id(target.kind),
                    0.035 / HALF_ARENA,
                    0.035 / HALF_ARENA,
                ],
                dtype=np.float32,
            )
            present[index] = float(target_present)
            visible[index] = float(target_visible)
            occluded[index] = float(target_present and not target_visible)

        for offset, (name, xy_array) in enumerate(sorted(env.pushable_obstacles.items())[:MAX_PUSHABLE_BOXES]):
            index = BOX_SLICE.start + offset
            xy = (float(xy_array[0]), float(xy_array[1]))
            box_visible = _visible_from_any_robot(env, xy, ignored_box=name)
            features[index] = np.asarray(
                [
                    xy[0] / HALF_ARENA,
                    xy[1] / HALF_ARENA,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    PUSHABLE_OBSTACLE_HALF / HALF_ARENA,
                    PUSHABLE_OBSTACLE_HALF / HALF_ARENA,
                ],
                dtype=np.float32,
            )
            present[index] = 1.0
            visible[index] = float(box_visible)
            occluded[index] = float(not box_visible)

        active = {
            (round(float(center[0]), 4), round(float(center[1]), 4), round(float(half[0]), 4), round(float(half[1]), 4))
            for center, half in active_base_armor_blockers(env.armor, inflated=False)
        }
        blocker_offset = 0
        for team in AGENTS:
            for center, size in BASE_ARMOR_SPECS[team]:
                if blocker_offset >= MAX_ARMOR_BLOCKERS:
                    break
                index = ARMOR_BLOCKER_SLICE.start + blocker_offset
                blocker_offset += 1
                half = (float(size[0]) * 0.5, float(size[1]) * 0.5)
                key = (round(float(center[0]), 4), round(float(center[1]), 4), round(half[0], 4), round(half[1], 4))
                blocker_present = key in active
                xy = (float(center[0]), float(center[1]))
                blocker_visible = _visible_from_any_robot(env, xy)
                features[index] = np.asarray(
                    [
                        xy[0] / HALF_ARENA,
                        xy[1] / HALF_ARENA,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        _team_id(team),
                        1.0,
                        half[0] / HALF_ARENA,
                        half[1] / HALF_ARENA,
                    ],
                    dtype=np.float32,
                )
                present[index] = float(blocker_present)
                visible[index] = float(blocker_visible)
                occluded[index] = float(blocker_present and not blocker_visible)

        if self.observation_dropout > 0.0:
            drop = self.rng.random(NUM_BELIEF_NODES) < self.observation_dropout
            drop[GLOBAL_SLICE] = False
            drop[ROBOT_SLICE] = False
            visible[drop] = 0.0
            occluded[drop & (present > 0.5)] = 1.0

        return _Measurement(features, present, visible, occluded, float(env.elapsed))

    def observe(self, env: RoboCupVisionRLSelfPlayEnv) -> BeliefGraph:
        current = self._measurement(env)
        self._history.append(current)
        if len(self._history) <= self.sensor_delay_steps:
            measured = _Measurement(
                current.features,
                current.present,
                np.zeros_like(current.visible),
                np.maximum(current.occluded, current.present),
                current.timestamp,
            )
            measured.visible[GLOBAL_SLICE] = 1.0
            measured.visible[ROBOT_SLICE] = current.visible[ROBOT_SLICE]
        else:
            measured = self._history.pop(0)
        now = float(env.elapsed)

        for index in range(NUM_BELIEF_NODES):
            is_visible = measured.visible[index] > 0.5
            previously_present = self.tokens[index, TOKEN_PRESENT] > 0.5
            node_type = NodeType(int(self.node_types[index]))
            if is_visible:
                previous_time = float(self.last_measurement_time[index])
                position = measured.features[index, :2]
                if np.isfinite(previous_time) and measured.timestamp > previous_time:
                    dt = max(measured.timestamp - previous_time, 1e-6)
                    measured.features[index, TOKEN_VX:TOKEN_VY + 1] = (position - self.last_positions[index]) / dt
                self.tokens[index, :PHYSICAL_TOKEN_DIM] = measured.features[index]
                self.last_positions[index] = position
                self.last_measurement_time[index] = measured.timestamp
                self.tokens[index, TOKEN_COVARIANCE] = 0.01
                self.tokens[index, TOKEN_PRESENT] = measured.present[index]
            elif not previously_present and measured.present[index] > 0.5:
                if node_type in (NodeType.TARGET, NodeType.ARMOR_BLOCKER):
                    # Field targets and armor geometry are map priors. Their live
                    # visibility and active state still come from measurements.
                    self.tokens[index, :PHYSICAL_TOKEN_DIM] = measured.features[index]
                self.tokens[index, TOKEN_PRESENT] = 1.0
                self.tokens[index, TOKEN_COVARIANCE] = 1.0

            last_seen = float(self.last_measurement_time[index])
            age_s = self.max_age_s if not np.isfinite(last_seen) else max(0.0, now - last_seen)
            self.tokens[index, TOKEN_VISIBLE] = float(is_visible)
            self.tokens[index, TOKEN_LAST_SEEN] = 0.0 if not np.isfinite(last_seen) else last_seen / max(float(env.max_time_s), 1e-6)
            self.tokens[index, TOKEN_AGE] = min(age_s / self.max_age_s, 1.0)
            if not is_visible:
                self.tokens[index, TOKEN_COVARIANCE] = min(
                    1.0,
                    float(self.tokens[index, TOKEN_COVARIANCE]) + self.covariance_growth * max(float(env.dt), 0.01),
                )
            self.tokens[index, TOKEN_OCCLUDED] = measured.occluded[index]

            event_synchronized = node_type in (NodeType.TARGET, NodeType.ARMOR_BLOCKER)
            if measured.present[index] <= 0.5 and (is_visible or event_synchronized):
                self.tokens[index, TOKEN_PRESENT] = 0.0
            if index == 0:
                self.tokens[index, TOKEN_PRESENT] = 1.0

        if not self.uncertainty_enabled:
            self.tokens[:, TOKEN_AGE] = 0.0
            self.tokens[:, TOKEN_COVARIANCE] = 0.0
            self.tokens[:, TOKEN_OCCLUDED] = 0.0

        return BeliefGraph(np.nan_to_num(self.tokens.copy()), self.node_types.copy())


def tokens_from_flat(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] != BELIEF_STATE_DIM:
        raise ValueError(f"belief state dim mismatch: {state.shape[-1]} != {BELIEF_STATE_DIM}")
    return state.reshape(*state.shape[:-1], NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)


def _point_segment_distance(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    segment = end - start
    denom = segment.square().sum(dim=-1).clamp_min(1e-8)
    alpha = ((points - start) * segment).sum(dim=-1) / denom
    alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
    projection = start + alpha * segment
    return torch.linalg.vector_norm(points - projection, dim=-1)


def build_typed_edges(tokens: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    """Construct sparse, typed interaction edges from current beliefs.

    Returns an adjacency tensor shaped ``[batch, edge_type, source, target]``.
    The construction depends only on token contents and types, so it remains
    equivariant to a joint permutation of tokens and type labels.
    """

    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if node_types.ndim == 1:
        node_types = node_types.unsqueeze(0).expand(tokens.shape[0], -1)
    batch, nodes, _features = tokens.shape
    present = tokens[..., TOKEN_PRESENT] > 0.5
    xy = tokens[..., TOKEN_X:TOKEN_Y + 1]
    pair_present = present[:, :, None] & present[:, None, :]
    global_node = (node_types == int(NodeType.GLOBAL)) & present
    robot = (node_types == int(NodeType.ROBOT)) & present
    target = (node_types == int(NodeType.TARGET)) & present
    box = (node_types == int(NodeType.BOX)) & present
    blocker = (node_types == int(NodeType.ARMOR_BLOCKER)) & present
    obstacle = box | blocker
    distance = torch.cdist(xy, xy)
    masks = torch.zeros(
        batch, NUM_EDGE_TYPES, nodes, nodes, dtype=torch.bool, device=tokens.device
    )

    masks[:, EdgeType.GLOBAL] = (
        (global_node[:, :, None] & present[:, None, :])
        | (present[:, :, None] & global_node[:, None, :])
    )
    masks[:, EdgeType.PROXIMITY] = pair_present & (distance < 0.42) & (distance > 0.0)
    masks[:, EdgeType.OBSERVES] = (
        robot[:, :, None]
        & present[:, None, :]
        & (tokens[..., TOKEN_VISIBLE] > 0.5)[:, None, :]
    )

    start = xy[:, :, None, None, :]
    end = xy[:, None, :, None, :]
    points = xy[:, None, None, :, :]
    segment = end - start
    denominator = segment.square().sum(dim=-1).clamp_min(1e-8)
    alpha = ((points - start) * segment).sum(dim=-1) / denominator
    projection = start + alpha.clamp(0.0, 1.0).unsqueeze(-1) * segment
    point_segment_distance = torch.linalg.vector_norm(points - projection, dim=-1)

    obstacle_clearance = tokens[..., TOKEN_EXTENT_X:TOKEN_EXTENT_Y + 1].amax(dim=-1).clamp_min(0.01)
    blocked_los = (
        (point_segment_distance <= obstacle_clearance[:, None, None, :])
        & obstacle[:, None, None, :]
    ).any(dim=-1)
    masks[:, EdgeType.LINE_OF_SIGHT] = (
        robot[:, :, None] & target[:, None, :] & ~blocked_los
    )

    contact_forward = (
        robot[:, :, None]
        & box[:, None, :]
        & (distance <= obstacle_clearance[:, None, :] + 0.12)
    )
    masks[:, EdgeType.CONTACTS] = contact_forward | contact_forward.transpose(1, 2)
    team = tokens[..., TOKEN_ATTRIBUTE_A]
    identity = torch.eye(nodes, dtype=torch.bool, device=tokens.device).unsqueeze(0)
    masks[:, EdgeType.THREATENS] = (
        robot[:, :, None]
        & robot[:, None, :]
        & ~identity
        & (team[:, :, None] * team[:, None, :] < 0.0)
        & (distance < 0.80)
    )

    box_blocks = (
        (point_segment_distance <= obstacle_clearance[:, None, None, :] + 0.025)
        & robot[:, :, None, None]
        & target[:, None, :, None]
        & box[:, None, None, :]
    ).any(dim=1)
    masks[:, EdgeType.BLOCKS_ROUTE] = box_blocks.transpose(1, 2)

    base_target = target & (tokens[..., TOKEN_ATTRIBUTE_B].abs() > 0.5)
    masks[:, EdgeType.PROTECTS_BASE] = (
        blocker[:, :, None]
        & base_target[:, None, :]
        & (team[:, :, None] * team[:, None, :] > 0.0)
        & (distance < 0.50)
    )
    return masks.to(tokens.dtype)


def extract_rule_risks(
    infos: dict[str, dict[str, object]],
    executed_actions: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    risks = np.zeros((len(AGENTS), NUM_RULE_RISKS), dtype=np.float32)
    own_fire_keys = (
        "own_target_hit",
        "own_base_hit",
        "own_target_blocked",
        "own_base_blocked",
    )
    for agent_index, team in enumerate(AGENTS):
        info = infos.get(team, {})
        risks[agent_index, RuleRisk.ROBOT_OR_TARGET_COLLISION] = float(
            bool(info.get("robot_contact")) or "target_collision" in info or bool(info.get("own_base_collision"))
        )
        risks[agent_index, RuleRisk.BLOCKED_OR_PENETRATION] = float(
            bool(info.get("blocked"))
            or bool(info.get("action_shield_contact"))
            or bool(info.get("bumper_or_hard_contact"))
        )
        risks[agent_index, RuleRisk.ILLEGAL_OR_OWN_TARGET_FIRE] = float(
            any(key in info for key in own_fire_keys) or bool(info.get("action_shield_fire"))
        )
        fire_requested = False
        if executed_actions is not None and team in executed_actions:
            fire_requested = float(np.asarray(executed_actions[team])[4]) > 0.55
        line_or_range_bad = bool(info.get("action_shield_fire")) or (
            fire_requested and not bool(info.get("line_clear", False))
        )
        risks[agent_index, RuleRisk.LOS_OR_RANGE_VIOLATION] = float(line_or_range_bad)
    return risks


def canonical_node_types_torch(batch_size: int, device: torch.device | str) -> torch.Tensor:
    values = torch.as_tensor(canonical_node_types(), dtype=torch.long, device=device)
    return values.unsqueeze(0).expand(int(batch_size), -1)
