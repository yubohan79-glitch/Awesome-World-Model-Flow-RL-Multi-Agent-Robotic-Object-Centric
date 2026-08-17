from __future__ import annotations

import math

import numpy as np

from ._compat import gym, spaces
from .datatypes import ShotResult, DomainRandomizationParams

from .constants import (
    AGENTS,
    BASE_RUSH_ARMOR_GATE,
    BASE_RUSH_BALANCED_NORMAL_HITS,
    BASE_RUSH_PREFERRED_NORMAL_HITS,
    DRAW_TIMEOUT_PENALTY,
    FIRE_YAW_TOLERANCE_RAD,
    RECOVERY_CONFIDENCE_THRESHOLD,
    SELFPLAY_OBSERVATION_DIM,
    SHOT_TIME_COST_SCALE,
    TACTICAL_ACTION_DIM
)
from robocup_visionrl_gym_env import (
    BLUE_BASE_XY,
    BLUE_START,
    LASER_FIRE_COOLDOWN_S,
    PUSHABLE_OBSTACLE_RANDOM_JITTER,
    PUSHABLE_OBSTACLE_STARTS,
    RoboCupVisionRLGymEnv,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    YELLOW_BASE_XY,
    YELLOW_START,
    shooting_range_limits
)


class RoboCupVisionRLSelfPlayEnvCore(gym.Env):
    metadata = {"render_modes": []}
    def __init__(
        self,
        dt: float = 0.10,
        max_time_s: float = 180.0,
        *,
        domain_randomization: bool = False,
        action_shield: bool = True,
    ):
        self.dt = dt
        self.max_time_s = max_time_s
        self.domain_randomization = bool(domain_randomization)
        self.action_shield = bool(action_shield)
        self.domain_params = DomainRandomizationParams()
        self.action_spaces = {
            team: spaces.Box(
                low=np.full(TACTICAL_ACTION_DIM, -1.0, dtype=np.float32),
                high=np.full(TACTICAL_ACTION_DIM, 1.0, dtype=np.float32),
                dtype=np.float32,
            )
            for team in AGENTS
        }
        self.observation_spaces = {
            team: spaces.Box(
                low=np.full(SELFPLAY_OBSERVATION_DIM, -np.inf, dtype=np.float32),
                high=np.full(SELFPLAY_OBSERVATION_DIM, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
            for team in AGENTS
        }
        helper = RoboCupVisionRLGymEnv(dt=dt, max_time_s=max_time_s)
        self.nav_blockers = helper.nav_blockers
        self.laser_blockers = helper.laser_blockers
        self.reset()
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.domain_params = self._sample_domain_randomization()
        noise_seed = int(seed if seed is not None else self.rng.integers(0, 2**31 - 1))
        first_noise = noise_seed * 2 + 101
        second_noise = noise_seed * 2 + 202
        first_base_noise = noise_seed * 2 + 303
        second_base_noise = noise_seed * 2 + 404
        if noise_seed % 2 == 0:
            self.shot_rng = {
                "yellow": np.random.default_rng(first_noise),
                "blue": np.random.default_rng(second_noise),
            }
            self.base_cap_rng = {
                "yellow": np.random.default_rng(first_base_noise),
                "blue": np.random.default_rng(second_base_noise),
            }
        else:
            self.shot_rng = {
                "yellow": np.random.default_rng(second_noise),
                "blue": np.random.default_rng(first_noise),
            }
            self.base_cap_rng = {
                "yellow": np.random.default_rng(second_base_noise),
                "blue": np.random.default_rng(first_base_noise),
            }
        self.base_rush_priority_team = AGENTS[noise_seed % len(AGENTS)]
        self.poses = {
            "yellow": YELLOW_START.copy(),
            "blue": BLUE_START.copy(),
        }
        self.targets = RoboCupVisionRLGymEnv()._make_targets()
        self.pushable_obstacles = self._sample_pushable_obstacle_starts()
        self.armor = {"yellow": 4, "blue": 4}
        self.scores = {"yellow": 0, "blue": 0}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.laser_locks = {
            team: {"target": "", "start": -99.0}
            for team in AGENTS
        }
        self.last_relocalization_time = {team: -999.0 for team in AGENTS}
        self.last_contact = False
        self.localization_confidence = {"yellow": 1.0, "blue": 1.0}
        self.sensor_fusion = {
            team: self._default_sensor_fusion_state()
            for team in AGENTS
        }
        self.winner: str | None = None
        self.pending_fire = {team: False for team in AGENTS}
        self.last_push_event = {team: "" for team in AGENTS}
        self.last_push_impulse = {team: {} for team in AGENTS}
        self.last_target_contact_time = {team: -99.0 for team in AGENTS}
        self.last_shot_attempt: dict[str, dict[str, float | str | bool]] = {team: {} for team in AGENTS}
        self.base_rush_lottery: dict[str, dict[tuple[str, int], bool]] = {team: {} for team in AGENTS}
        self.base_retry_min_normal_hits = {team: 0 for team in AGENTS}
        self.selected_target_name: dict[str, str | None] = {team: None for team in AGENTS}
        self.last_motion_command = {team: (0.0, 0.0) for team in AGENTS}
        self.target_order: dict[str, list[str]] = {team: [] for team in AGENTS}
        self.target_fail_counts: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.target_cooldowns: dict[str, dict[str, float]] = {team: {} for team in AGENTS}
        self.lost_targets: dict[str, set[str]] = {team: set() for team in AGENTS}
        self.fire_pose_blocked_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.normal_attack_stale_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.base_attack_stale_steps: dict[str, dict[str, int]] = {team: {} for team in AGENTS}
        self.post_hit_retreat_until = {team: -99.0 for team in AGENTS}
        self._fire_pose_cache: dict[tuple[str, int], list[tuple[np.ndarray, float, float, float]]] = {}
        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]], list[np.ndarray]] = {}
        self._route_distance_cache: dict[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]], float] = {}
        self.strategy_counts = {
            team: {
                "attack_steps": 0,
                "base_rush_steps": 0,
                "block_steps": 0,
                "interference_steps": 0,
                "recovery_steps": 0,
                "normal_hits": 0,
                "base_hits": 0,
            }
            for team in AGENTS
        }
        self.previous_base_distance = {
            "yellow": float(np.linalg.norm(self.poses["yellow"][:2] - BLUE_BASE_XY)),
            "blue": float(np.linalg.norm(self.poses["blue"][:2] - YELLOW_BASE_XY)),
        }
        self.previous_attack_distance = {
            team: self._nearest_opponent_target_distance(team)
            for team in AGENTS
        }
        return {team: self._obs(team) for team in AGENTS}, {team: {} for team in AGENTS}
    def step(self, actions: dict[str, np.ndarray]):
        rewards = {team: -0.018 for team in AGENTS}
        infos: dict[str, dict[str, object]] = {team: {} for team in AGENTS}
        action_values = {
            team: self._coerce_action(actions.get(team, np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)))
            for team in AGENTS
        }
        team_order = list(AGENTS)
        if bool(self.rng.integers(0, 2)):
            team_order.reverse()
        self.elapsed += self.dt
        for team in team_order:
            previous = self._base_distance(team)
            previous_attack_distance = self._nearest_opponent_target_distance(team)
            action = action_values[team]
            blocked, decision_info = self._apply_action(team, action)
            decision_info["step_order"] = team_order.index(team)
            infos[team].update(decision_info)
            rewards[team] += 0.10 * (previous - self._base_distance(team))
            rewards[team] += 0.13 * (previous_attack_distance - self._nearest_opponent_target_distance(team))
            if decision_info.get("tactic") == "block":
                threat = float(decision_info.get("opponent_threat", 0.0))
                rewards[team] += -0.045 + 0.08 * threat
            if decision_info.get("tactic") == "recover":
                useful = self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD
                rewards[team] += 0.055 if useful else -0.08
            if decision_info.get("base_rush") and self.armor[self._opponent(team)] > BASE_RUSH_ARMOR_GATE:
                rewards[team] -= 0.10
            if decision_info.get("base_rush"):
                normal_hits = self._normal_hits_against(team)
                if normal_hits <= 1:
                    rewards[team] -= 0.18
                elif normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS:
                    rewards[team] -= 0.055
                elif normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                    rewards[team] += 0.095
            if decision_info.get("tactic") == "attack":
                goal_distance = float(decision_info.get("goal_distance_m", 1.0))
                rewards[team] += 0.06 * max(0.0, 1.0 - goal_distance / 0.75)
                shot_distance = float(decision_info.get("shot_distance_m", SHOOT_RANGE))
                close_quality = self._shot_accuracy_from_geometry(
                    shot_distance,
                    float(decision_info.get("shot_lateral_error_m", SHOOT_HIT_RADIUS)),
                    bool(decision_info.get("base_rush", False)),
                )
                rewards[team] += 0.09 * close_quality
                rewards[team] -= SHOT_TIME_COST_SCALE * max(0.0, 0.38 - shot_distance)
                fire_requested = float(action[4]) > 0.55
                if decision_info.get("fire_ready"):
                    base_bonus = 0.28 if decision_info.get("base_rush") else 0.0
                    rewards[team] += 0.24 + 0.30 * close_quality + base_bonus
                elif fire_requested:
                    distance = float(decision_info.get("shot_distance_m", 9.0))
                    yaw_error = float(decision_info.get("shot_yaw_error_rad", math.pi))
                    miss_penalty = 0.06
                    min_range, max_range = shooting_range_limits(bool(decision_info.get("base_rush", False)))
                    if distance > max_range or distance < min_range:
                        miss_penalty += 0.08
                    if yaw_error > FIRE_YAW_TOLERANCE_RAD:
                        miss_penalty += 0.05
                    rewards[team] -= miss_penalty
            if decision_info.get("pushed_obstacle"):
                useful_push = decision_info.get("tactic") in ("attack", "push_clear")
                rewards[team] += 0.18 if useful_push else -0.05
            if blocked:
                rewards[team] -= 0.10
                infos[team]["blocked"] = True
        if self._resolve_contact():
            for team in AGENTS:
                contact_reward = self._contact_reward(team, infos[team])
                rewards[team] += contact_reward
                infos[team]["robot_contact"] = True
                infos[team]["tactical_contact"] = contact_reward > 0.0
        for team in AGENTS:
            self._resolve_target_contacts(team, rewards, infos)
        for team in AGENTS:
            action = action_values[team]
            if self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD:
                recovery_requested = bool(infos[team].get("tactic") == "recover") or float(action[3]) > 0.35
                if recovery_requested and self._can_relocalize(team):
                    self.localization_confidence[team] = min(1.0, self.localization_confidence[team] + 0.34)
                    self.last_relocalization_time[team] = self.elapsed
                    self._boost_sensor_fusion_recovery(team)
                    rewards[team] += 0.08
                    infos[team]["relocalizing"] = True
                elif recovery_requested:
                    rewards[team] -= 0.02
                    infos[team]["relocalization_cooldown"] = True
                else:
                    rewards[team] -= 0.14
        shot_results: list[ShotResult] = []
        missed_shots: list[str] = []
        for team in team_order:
            if self.pending_fire[team] and self.elapsed - self.last_fire[team] > LASER_FIRE_COOLDOWN_S:
                result = self._apply_fire(team)
                if result is not None:
                    self.last_fire[team] = self.elapsed
                    shot_results.append(result)
                else:
                    reason = str(self.last_shot_attempt.get(team, {}).get("reason", ""))
                    if reason == "dwell":
                        rewards[team] += 0.018
                    elif reason == "probabilistic_miss":
                        self.last_fire[team] = self.elapsed
                        infos[team]["shot_attempt"] = dict(self.last_shot_attempt[team])
                        rewards[team] -= 0.025
                    else:
                        missed_shots.append(team)
            else:
                self._reset_laser_lock(team)
        base_win_results = [
            result for result in shot_results
            if result.kind == f"base_{self._opponent(result.shooter)}"
        ]
        if len(base_win_results) == 2:
            for result in shot_results:
                self._score_shot(result, rewards, infos, terminal_override=False)
            self.winner = "draw"
            for result in base_win_results:
                infos[result.shooter]["simultaneous_base_hit"] = True
        else:
            for result in shot_results:
                self._score_shot(result, rewards, infos)
        for team in missed_shots:
            target_name = self.selected_target_name.get(team)
            shot_attempt = self.last_shot_attempt.get(team, {})
            if target_name and str(target_name).endswith("BaseTarget") and shot_attempt.get("reason") == "base_cap_failed":
                # A failed base cap lottery means this normal-hit bucket cannot
                # legally win an early base in this episode. Force one- and
                # two-hit rushes to improve the success window, but allow a
                # three-hit attack to retry after cooldown because it is already
                # the intended high-probability tempo.
                normal_hits = self._normal_hits_against(team)
                required_hits = min(4, normal_hits + 1) if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS else normal_hits
                self.base_retry_min_normal_hits[team] = max(
                    int(self.base_retry_min_normal_hits.get(team, 0)),
                    required_hits,
                )
                self._fire_pose_cache.clear()
            if target_name:
                self._mark_target_failed(team, target_name)
            infos[team]["shot_attempt"] = dict(self.last_shot_attempt[team])
            rewards[team] -= 0.08
        terminated = self.winner is not None
        truncated = self.elapsed >= self.max_time_s
        if truncated and self.winner is None:
            if self.scores["yellow"] > self.scores["blue"]:
                self.winner = "yellow"
            elif self.scores["blue"] > self.scores["yellow"]:
                self.winner = "blue"
            else:
                self.winner = "draw"
        if self.winner in AGENTS:
            loser = self._opponent(self.winner)
            rewards[self.winner] += 80.0
            rewards[loser] -= 55.0
        elif self.winner == "draw":
            for team in AGENTS:
                rewards[team] -= DRAW_TIMEOUT_PENALTY
        observations = {team: self._obs(team) for team in AGENTS}
        terminations = {team: terminated for team in AGENTS}
        truncations = {team: truncated for team in AGENTS}
        return observations, rewards, terminations, truncations, infos
    def _coerce_action(self, action: np.ndarray | None) -> np.ndarray:
        values = np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32)
        if action is not None:
            raw = np.asarray(action, dtype=np.float32).reshape(-1)
            count = min(raw.shape[0], TACTICAL_ACTION_DIM)
            values[:count] = raw[:count]
        return np.clip(values, -1.0, 1.0)
    def _sample_domain_randomization(self) -> DomainRandomizationParams:
        if not self.domain_randomization:
            return DomainRandomizationParams()
        return DomainRandomizationParams(
            drive_scale=float(self.rng.uniform(0.92, 1.08)),
            turn_scale=float(self.rng.uniform(0.90, 1.10)),
            push_step_scale=float(self.rng.uniform(0.72, 1.18)),
            shot_accuracy_scale=float(self.rng.uniform(0.82, 1.05)),
            drift_loss_scale=float(self.rng.uniform(0.85, 1.40)),
            sensor_noise_scale=float(self.rng.uniform(0.0, 0.035)),
        )
    def _sample_pushable_obstacle_starts(self) -> dict[str, np.ndarray]:
        starts = {name: value.copy() for name, value in PUSHABLE_OBSTACLE_STARTS.items()}
        if not self.domain_randomization:
            return starts
        for name, base_xy in starts.items():
            jitter = self.rng.uniform(-PUSHABLE_OBSTACLE_RANDOM_JITTER, PUSHABLE_OBSTACLE_RANDOM_JITTER, size=2)
            candidate = base_xy + jitter.astype(np.float32)
            sign = np.sign(base_xy)
            lower = np.array([0.58, 0.58], dtype=np.float32) * sign
            upper = np.array([0.96, 0.96], dtype=np.float32) * sign
            starts[name] = np.minimum(np.maximum(candidate, np.minimum(lower, upper)), np.maximum(lower, upper))
        return starts
    def _shield_contact_action(self, team: str, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.action_shield:
            return action, False
        if not self._near_own_critical_assets(team):
            return action, False
        shielded = action.copy()
        changed = False
        if float(shielded[2]) > 0.10:
            shielded[2] = min(float(shielded[2]), -0.25)
            changed = True
        if float(shielded[5]) > 0.35:
            shielded[5] = min(float(shielded[5]), 0.20)
            changed = True
        return shielded, changed
