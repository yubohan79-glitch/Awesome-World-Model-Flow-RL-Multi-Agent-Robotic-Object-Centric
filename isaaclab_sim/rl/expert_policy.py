from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robocup_visionrl_selfplay_env import (
    AGENTS,
    BASE_RUSH_BALANCED_NORMAL_HITS,
    BASE_RUSH_EARLY_NORMAL_HITS,
    BASE_RUSH_PREFERRED_NORMAL_HITS,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    MIN_SHOOT_DISTANCE,
    RECOVERY_CONFIDENCE_THRESHOLD,
    SHOOT_RANGE,
    TACTICAL_ACTION_DIM,
    RoboCupVisionRLSelfPlayEnv,
    laser_origin_from_pose,
    wrap_angle,
)


@dataclass(frozen=True)
class TeamExpertProfile:
    name: str
    normal_order: tuple[str, ...]
    side_gate_targets: tuple[str, ...]
    post_base_retry_order: tuple[str, ...]
    one_hit_base_gate: float
    two_hit_base_gate: float
    three_hit_base_gate: float
    default_risk: float
    urgent_risk: float
    base_risk: float
    push_risk_bonus: float
    block_bias: float
    target_bias: dict[str, float]


YELLOW_EXPERT = TeamExpertProfile(
    name="yellow_expert",
    # Yellow starts in the south-east lane. It first takes the north-middle
    # shot on the right lane, then turns to the west side-gate base window.
    normal_order=("T01_NorthMiddle", "T03_WestAboveGate", "T05_EastAboveGate", "T02_NorthEast"),
    side_gate_targets=("T03_WestAboveGate", "T05_EastAboveGate"),
    post_base_retry_order=("T05_EastAboveGate", "T02_NorthEast"),
    one_hit_base_gate=0.82,
    two_hit_base_gate=0.58,
    three_hit_base_gate=0.92,
    default_risk=0.34,
    urgent_risk=0.62,
    base_risk=0.76,
    push_risk_bonus=0.10,
    block_bias=0.00,
    target_bias={
        "T01_NorthMiddle": 0.58,
        "T03_WestAboveGate": 0.42,
        "T05_EastAboveGate": 0.18,
        "T02_NorthEast": 0.22,
    },
)

BLUE_EXPERT = TeamExpertProfile(
    name="blue_expert",
    # Blue mirrors the task with a south-middle opener, then turns to the east
    # side-gate window before deciding whether a two-hit base rush is worth it.
    normal_order=("T08_SouthMiddle", "T06_EastBelowGate", "T04_WestBelowGate", "T07_SouthWest"),
    side_gate_targets=("T06_EastBelowGate", "T04_WestBelowGate"),
    post_base_retry_order=("T04_WestBelowGate", "T07_SouthWest"),
    one_hit_base_gate=0.86,
    two_hit_base_gate=0.58,
    three_hit_base_gate=0.94,
    default_risk=0.38,
    urgent_risk=0.66,
    base_risk=0.75,
    push_risk_bonus=0.13,
    block_bias=0.04,
    target_bias={
        "T08_SouthMiddle": 0.26,
        "T06_EastBelowGate": 0.42,
        "T04_WestBelowGate": 0.18,
        "T07_SouthWest": 0.24,
    },
)

TEAM_EXPERTS = {
    "yellow": YELLOW_EXPERT,
    "blue": BLUE_EXPERT,
}


def opponent(team: str) -> str:
    return "blue" if team == "yellow" else "yellow"


def expert_profile(team: str) -> TeamExpertProfile:
    try:
        return TEAM_EXPERTS[team]
    except KeyError as exc:
        raise ValueError(f"unknown team for expert profile: {team!r}") from exc


def _profile_target_bias(profile: TeamExpertProfile, target_name: str, normal_hits: int) -> float:
    bias = float(profile.target_bias.get(target_name, 0.0))
    if normal_hits == 0:
        try:
            index = profile.normal_order.index(target_name)
        except ValueError:
            return bias
        return bias + max(0.0, 0.38 - 0.11 * index)
    if normal_hits == 1 and target_name in profile.side_gate_targets:
        return bias + 0.42
    if target_name in profile.normal_order:
        remaining_order_bonus = max(0.0, 0.24 - 0.055 * profile.normal_order.index(target_name))
        return bias + remaining_order_bonus
    return bias


def _base_commit_allowed(
    profile: TeamExpertProfile,
    team: str,
    *,
    normal_hits: int,
    score_delta: float,
    time_ratio: float,
    risk: float,
    priority_team: str | None = None,
) -> bool:
    if normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
        return True
    if normal_hits < BASE_RUSH_EARLY_NORMAL_HITS:
        return False
    behind_or_late = score_delta < -5 or time_ratio > 0.74
    if normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
        if priority_team in AGENTS and team != priority_team and not behind_or_late:
            return False
        return risk > 0.64 or behind_or_late
    return risk > 0.88 and behind_or_late


def select_target(
    env: RoboCupVisionRLSelfPlayEnv,
    team: str,
    risk: float = 0.62,
    profile: TeamExpertProfile | None = None,
):
    """Pick the nearest legal opponent target with a valid firing pose.

    The expert attacks opponent targets only. Yellow and blue use separate
    profiles so their opening order, push willingness, and base-rush tempo can
    diverge before the residual actors are trained.
    """

    profile = expert_profile(team) if profile is None else profile
    other = opponent(team)
    normal_targets = [
        target
        for target in env.targets
        if target.kind == "normal" and target.owner == other and not target.knocked
    ]
    base_targets = [
        target
        for target in env.targets
        if target.kind == f"base_{other}" and not target.knocked
    ]

    normal_hits = env._normal_hits_against(team)
    score_delta = env.scores[team] - env.scores[other]
    time_ratio = env.elapsed / max(env.max_time_s, 1e-6)
    retry_min_hits = int(getattr(env, "base_retry_min_normal_hits", {}).get(team, 0))
    if normal_hits < retry_min_hits:
        for name in profile.post_base_retry_order:
            target = next((item for item in normal_targets if item.name == name), None)
            if target is not None and env._best_fire_pose(team, target, risk=min(0.96, risk + 0.16), route_aware=True) is not None:
                return target
    if normal_hits == 0:
        for name in profile.normal_order[:2]:
            target = next((item for item in normal_targets if item.name == name), None)
            if target is not None and env._best_fire_pose(team, target, risk=risk, route_aware=True) is not None:
                return target
    if normal_hits == 1:
        for name in profile.side_gate_targets:
            target = next((item for item in normal_targets if item.name == name), None)
            if target is not None and env._best_fire_pose(team, target, risk=risk, route_aware=True) is not None:
                return target
    if normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS and base_targets:
        reachable_base = []
        for target in base_targets:
            solution = env._best_fire_pose(team, target, risk=min(0.92, risk + 0.18), route_aware=True)
            if solution is None:
                continue
            _fire_xy, _route_distance, _shot_distance, shot_quality = solution
            quality_gate = {1: 0.30, 2: 0.24, 3: 0.13, 4: 0.04}[max(1, min(4, normal_hits))]
            if profile.name == "blue_expert" and normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS:
                quality_gate = 0.18
            if shot_quality >= quality_gate:
                reachable_base.append(target)
        if reachable_base and _base_commit_allowed(
            profile,
            team,
            normal_hits=normal_hits,
            score_delta=score_delta,
            time_ratio=time_ratio,
            risk=risk,
            priority_team=getattr(env, "base_rush_priority_team", None),
        ):
            return reachable_base[0]

    candidates = normal_targets + ([] if normal_targets else base_targets)
    if not candidates:
        return None

    scored = []
    pose = env.poses[team]
    for target in candidates:
        solution = env._best_fire_pose(team, target, risk=risk, route_aware=True)
        if solution is None:
            continue
        fire_xy, route_distance, shot_distance, shot_quality = solution
        time_cost = 0.35 * route_distance
        precision_cost = 0.42 * abs(shot_distance - 0.30)
        blocker_cost = 0.18 * float(np.linalg.norm(fire_xy - pose[:2]))
        quota_cost = 0.0
        quota_cost -= _profile_target_bias(profile, target.name, normal_hits)
        if target.kind == "normal" and normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
            quota_cost += 0.34
        if target.kind == "normal" and normal_hits >= 4:
            quota_cost += 0.62
        if target.kind.startswith("base_") and normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS:
            if normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                quota_cost -= 0.24
            elif normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
                quota_cost += 0.12
            else:
                quota_cost += 0.42
        scored.append((time_cost + precision_cost + blocker_cost + quota_cost - 0.42 * shot_quality, target))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def _fire_readiness(env: RoboCupVisionRLSelfPlayEnv, team: str, target) -> tuple[bool, float, float]:
    geometry = env._fire_geometry_snapshot(team, target, risk=0.70)
    pose = env.poses[team]
    origin = laser_origin_from_pose(pose)
    dx = float(target.xy[0] - origin[0])
    dy = float(target.xy[1] - origin[1])
    forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
    shot_distance = dx * forward[0] + dy * forward[1]
    bearing = math.atan2(dy, dx)
    yaw_error = abs(wrap_angle(bearing - float(pose[2])))
    ready = bool(geometry["geometry_ready"])
    return ready, shot_distance, yaw_error


def _selector_for_requested_target(
    env: RoboCupVisionRLSelfPlayEnv,
    team: str,
    requested_name: str,
    action: np.ndarray,
    risk: float,
) -> float:
    opponent_team = opponent(team)
    env._refresh_target_visibility_memory(team)
    normal_targets = [
        target
        for target in env.targets
        if target.owner == opponent_team
        and not target.knocked
        and not env._target_on_cooldown(team, target.name)
        and target.kind == "normal"
    ]
    base_targets = [
        target
        for target in env.targets
        if target.kind == f"base_{opponent_team}"
        and not target.knocked
        and not env._target_on_cooldown(team, target.name)
    ]
    normal_hits = env._normal_hits_against(team)
    time_remaining = max(0.0, env.max_time_s - env.elapsed)
    low_time = time_remaining < 45.0
    base_gate = float(action[1])
    allow_base = (
        (
            normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS
            and base_gate > 0.22
            and risk > 0.50
        )
        or env._base_rush_open(team, action)
        or low_time
    )
    candidates = list(normal_targets)
    if allow_base or not candidates:
        candidates.extend(base_targets)
    if not candidates:
        return 0.0
    scored = [
        (env._target_priority(team, target, action, risk, route_aware=True), target)
        for target in candidates
    ]
    ranked = [
        target for priority, target in sorted(scored, key=lambda item: item[0], reverse=True)
        if priority > -50.0
    ]
    if not ranked:
        return 0.0
    near_window = min(3, len(ranked))
    for index, target in enumerate(ranked[:near_window]):
        if target.name == requested_name:
            return float((index / max(1, near_window - 1)) * 2.0 - 1.0)
    return 0.96 if requested_name.endswith("BaseTarget") else 1.0


def _expert_action(env: RoboCupVisionRLSelfPlayEnv, team: str, profile: TeamExpertProfile) -> np.ndarray:
    if env.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD:
        fusion = env.sensor_fusion.get(team, {})
        hard_contact = float(fusion.get("bumper_or_hard_contact", 0.0)) > 0.5
        scan_bad = float(fusion.get("scan_clearance", 1.0)) < 0.18
        odom_bad = float(fusion.get("wheel_imu_consistency", 1.0)) < 0.22
        critical_confidence = env.localization_confidence[team] < max(0.18, RECOVERY_CONFIDENCE_THRESHOLD - 0.08)
        if hard_contact or (critical_confidence and (scan_bad or odom_bad)):
            spin_sign = -0.65 if team == "yellow" else 0.65
            return np.array([0.0, -0.35, -0.85, 0.95, -0.65, spin_sign], dtype=np.float32)

    other = opponent(team)
    score_delta = env.scores[team] - env.scores[other]
    time_ratio = env.elapsed / max(env.max_time_s, 1e-6)
    urgent = score_delta < 0 or time_ratio > 0.72
    attack_risk = profile.default_risk if not urgent else profile.urgent_risk
    normal_hits = env._normal_hits_against(team)
    if normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS:
        attack_risk = max(attack_risk, 0.46)
    if normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
        attack_risk = max(attack_risk, 0.56 + profile.push_risk_bonus * 0.35)
    if normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
        attack_risk = max(attack_risk, 0.66 + profile.push_risk_bonus * 0.25)
    retry_min_hits = int(getattr(env, "base_retry_min_normal_hits", {}).get(team, 0))
    if normal_hits < retry_min_hits:
        attack_risk = max(attack_risk, 0.76 + profile.push_risk_bonus * 0.50)

    target = select_target(env, team, risk=(attack_risk + 1.0) * 0.5, profile=profile)
    if target is None:
        return np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)

    is_base = target.kind.startswith("base_")
    own_base = np.array([1.25, -1.25], dtype=np.float32) if team == "yellow" else np.array([-1.25, 1.25], dtype=np.float32)
    opponent_to_own_base = float(np.linalg.norm(env.poses[other][:2] - own_base))
    ready, shot_distance, _yaw_error = _fire_readiness(env, team, target)

    if is_base or normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
        base_rush_gate = profile.three_hit_base_gate
    elif normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
        base_rush_gate = min(0.94, profile.two_hit_base_gate + (0.10 if urgent else 0.0))
    elif normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS:
        base_rush_gate = profile.one_hit_base_gate if urgent and attack_risk > 0.76 else -0.18
    else:
        base_rush_gate = -0.72
    block_gate = -0.28 + profile.block_bias
    if (score_delta >= 10 and time_ratio > 0.68) or opponent_to_own_base < 0.72:
        block_gate = 0.62 + profile.block_bias

    recovery_gate = -0.64
    fire_gate = 0.95 if ready else -0.72
    min_range = BASE_SHOOT_MIN_RANGE if is_base else MIN_SHOOT_DISTANCE
    max_range = BASE_SHOOT_RANGE if is_base else SHOOT_RANGE
    if shot_distance < min_range - 0.02 or shot_distance > max_range + 0.02:
        fire_gate = -0.18

    if is_base:
        attack_risk = max(attack_risk, profile.base_risk)
    if env.sensor_fusion[team].get("pushable_contact", 0.0) > 0.5:
        attack_risk = min(0.96, attack_risk + profile.push_risk_bonus)
    action = np.array(
        [
            0.0,
            base_rush_gate,
            block_gate,
            recovery_gate,
            fire_gate,
            attack_risk,
        ],
        dtype=np.float32,
    )
    action[0] = _selector_for_requested_target(env, team, target.name, action, (attack_risk + 1.0) * 0.5)
    return action


def yellow_expert_action(env: RoboCupVisionRLSelfPlayEnv) -> np.ndarray:
    return _expert_action(env, "yellow", YELLOW_EXPERT)


def blue_expert_action(env: RoboCupVisionRLSelfPlayEnv) -> np.ndarray:
    return _expert_action(env, "blue", BLUE_EXPERT)


def scripted_action(env: RoboCupVisionRLSelfPlayEnv, team: str) -> np.ndarray:
    if team == "yellow":
        return yellow_expert_action(env)
    if team == "blue":
        return blue_expert_action(env)
    raise ValueError(f"unknown team for scripted action: {team!r}")


def residual_expert_action(
    env: RoboCupVisionRLSelfPlayEnv,
    team: str,
    residual: np.ndarray,
    *,
    residual_scale: float = 0.28,
) -> np.ndarray:
    """Blend an RL residual with the safe scripted policy.

    The residual can change tactical preference and timing but is bounded per
    dimension so it cannot easily disable recovery or violate the opponent-only
    shooting gate that is enforced by the environment.
    """

    base = scripted_action(env, team)
    residual = np.asarray(residual, dtype=np.float32).reshape(-1)
    padded = np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)
    padded[: min(TACTICAL_ACTION_DIM, residual.shape[0])] = residual[:TACTICAL_ACTION_DIM]
    dimension_scale = np.array([0.38, 0.34, 1.25, 0.22, 0.30, 0.80], dtype=np.float32)
    return np.clip(base + float(residual_scale) * dimension_scale * padded, -1.0, 1.0).astype(np.float32)


def compose_policy_action(
    env: RoboCupVisionRLSelfPlayEnv,
    team: str,
    model_action: np.ndarray,
    *,
    policy_mode: str = "direct",
    residual_scale: float = 0.28,
) -> np.ndarray:
    if policy_mode == "direct":
        return np.clip(np.asarray(model_action, dtype=np.float32), -1.0, 1.0)
    if policy_mode == "expert":
        return scripted_action(env, team)
    if policy_mode == "residual_expert":
        return residual_expert_action(env, team, model_action, residual_scale=residual_scale)
    raise ValueError(f"unknown policy_mode: {policy_mode}")


def batched_actions_to_env(
    envs: list[RoboCupVisionRLSelfPlayEnv],
    model_actions: np.ndarray,
    *,
    policy_mode: str = "direct",
    residual_scale: float = 0.28,
) -> list[dict[str, np.ndarray]]:
    clipped = np.clip(model_actions, -1.0, 1.0).astype(np.float32)
    env_actions: list[dict[str, np.ndarray]] = []
    cursor = 0
    for env in envs:
        action_dict: dict[str, np.ndarray] = {}
        for team in AGENTS:
            action_dict[team] = compose_policy_action(
                env,
                team,
                clipped[cursor],
                policy_mode=policy_mode,
                residual_scale=residual_scale,
            )
            cursor += 1
        env_actions.append(action_dict)
    return env_actions
