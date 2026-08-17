from __future__ import annotations

import math

import numpy as np

from .constants import (
    BASE_RUSH_BALANCED_NORMAL_HITS,
    BASE_RUSH_EARLY_NORMAL_HITS,
    BASE_RUSH_PREFERRED_NORMAL_HITS,
    POST_HIT_RETREAT_S
)
from .datatypes import ShotResult
from .geometry import (
    laser_origin_from_pose
)
from robocup_visionrl_gym_env import (
    BASE_HIT_RADIUS,
    BASE_SHOOT_RANGE,
    LASER_DWELL_REQUIRED_S,
    PUSHABLE_OBSTACLE_HALF,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    Target,
    active_base_armor_blockers,
    normalized_laser_dwell_factor,
    segment_intersects_aabb,
    shooting_range_limits
)


class LaserMixin:
    def _apply_fire(self, team: str) -> ShotResult | None:
        target = self._detect_laser_hit(team)
        if target is None:
            return None
        return ShotResult(team, target.name, target.owner, target.kind)
    def _update_laser_lock(self, team: str, target_name: str) -> float:
        lock = self.laser_locks[team]
        if lock["target"] != target_name:
            lock["target"] = target_name
            lock["start"] = self.elapsed
            return 0.0
        return max(0.0, self.elapsed - float(lock["start"]))
    def _reset_laser_lock(self, team: str):
        self.laser_locks[team]["target"] = ""
        self.laser_locks[team]["start"] = -99.0
    def _detect_laser_hit(self, team: str) -> Target | None:
        pose = self.poses[team]
        origin = laser_origin_from_pose(pose)
        forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        best_target = None
        best_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0
        best_accuracy = 0.0
        best_lateral_error = 0.0
        own_candidate_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0
        for target in self.targets:
            if target.knocked:
                continue
            if (
                target.kind.startswith("base_")
                and target.owner != team
                and self._normal_hits_against(team) < BASE_RUSH_EARLY_NORMAL_HITS
            ):
                continue
            dx = target.xy[0] - origin[0]
            dy = target.xy[1] - origin[1]
            projection = dx * forward[0] + dy * forward[1]
            min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
            if projection < min_range or projection > max_range:
                continue
            hit_radius = BASE_HIT_RADIUS if target.kind.startswith("base_") else SHOOT_HIT_RADIUS
            perpendicular = abs(dx * forward[1] - dy * forward[0])
            if perpendicular > hit_radius:
                continue
            if not self._target_line_clear(team, origin, target):
                continue
            if target.owner == team:
                own_candidate_projection = min(own_candidate_projection, projection)
                continue
            accuracy = self._shot_accuracy_from_geometry(projection, perpendicular, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                pose_quality = self._base_attack_pose_quality(team, target, pose[:2])
                if pose_quality <= 0.0:
                    continue
                accuracy *= pose_quality
            if projection < best_projection:
                best_projection = projection
                best_target = target
                best_accuracy = accuracy
                best_lateral_error = perpendicular
        if own_candidate_projection <= max(SHOOT_RANGE, BASE_SHOOT_RANGE) and own_candidate_projection <= best_projection:
            self._reset_laser_lock(team)
            self.last_shot_attempt[team] = {"hit": False, "reason": "own_target_safety_gate"}
            return None
        if best_target is None:
            self._reset_laser_lock(team)
            self.last_shot_attempt[team] = {"hit": False, "reason": "no_geometry"}
            return None
        dwell_s = self._update_laser_lock(team, best_target.name)
        if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
            self.last_shot_attempt[team] = {
                "hit": False,
                "reason": "dwell",
                "target": best_target.name,
                "dwell_s": round(float(dwell_s), 3),
                "required_s": LASER_DWELL_REQUIRED_S,
                "distance_m": round(float(best_projection), 4),
                "lateral_error_m": round(float(best_lateral_error), 4),
                "accuracy": round(float(best_accuracy), 4),
            }
            return None
        dwell_factor = normalized_laser_dwell_factor(dwell_s)
        final_accuracy = float(
            np.clip(best_accuracy * dwell_factor * self.domain_params.shot_accuracy_scale, 0.0, 0.95)
        )
        if best_target.kind.startswith("base_"):
            normal_hits = max(0, min(4, self._normal_hits_against(team)))
            if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS:
                lottery_key = (best_target.name, normal_hits)
                if lottery_key not in self.base_rush_lottery[team]:
                    self.base_rush_lottery[team][lottery_key] = bool(
                        self.base_cap_rng[team].random() <= self._base_hit_cap_for_team(team)
                    )
                base_cap_passed = self.base_rush_lottery[team][lottery_key]
            else:
                base_cap_passed = bool(self.base_cap_rng[team].random() <= self._base_hit_cap_for_team(team))
            if not base_cap_passed:
                self.target_cooldowns[team][best_target.name] = max(
                    self.target_cooldowns[team].get(best_target.name, -99.0),
                    self.elapsed + 14.0,
                )
                self.last_shot_attempt[team] = {
                    "hit": False,
                    "reason": "base_cap_failed",
                    "target": best_target.name,
                    "dwell_s": round(float(dwell_s), 3),
                    "distance_m": round(float(best_projection), 4),
                    "lateral_error_m": round(float(best_lateral_error), 4),
                    "geometry_accuracy": round(float(best_accuracy), 4),
                    "dwell_factor": round(float(dwell_factor), 4),
                    "accuracy": 0.0,
                    "base_hit_cap": round(float(self._base_hit_cap_for_team(team)), 4),
                }
                self._reset_laser_lock(team)
                return None
        hit = bool(self.shot_rng[team].random() <= final_accuracy)
        self.last_shot_attempt[team] = {
            "hit": hit,
            "reason": "" if hit else "probabilistic_miss",
            "target": best_target.name,
            "dwell_s": round(float(dwell_s), 3),
            "distance_m": round(float(best_projection), 4),
            "lateral_error_m": round(float(best_lateral_error), 4),
            "geometry_accuracy": round(float(best_accuracy), 4),
            "dwell_factor": round(float(dwell_factor), 4),
            "accuracy": round(float(final_accuracy), 4),
            "base_hit_cap": round(float(self._base_hit_cap_for_team(team)), 4)
            if best_target.kind.startswith("base_")
            else "",
        }
        if hit:
            self._reset_laser_lock(team)
            return best_target
        return None
    def _line_blocked(self, origin: tuple[float, float], target_xy: tuple[float, float]) -> bool:
        for center, half_size in self.laser_blockers:
            if segment_intersects_aabb(origin, target_xy, center, half_size):
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            if segment_intersects_aabb(origin, target_xy, center, half_size):
                return True
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(
                origin,
                target_xy,
                (float(center[0]), float(center[1])),
                (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
            ):
                return True
        return False
    def _score_shot(
        self,
        result: ShotResult,
        rewards: dict[str, float],
        infos: dict[str, dict[str, object]],
        *,
        terminal_override: bool = True,
    ):
        shooter = result.shooter
        opponent = self._opponent(shooter)
        target = next(t for t in self.targets if t.name == result.target_name)
        if result.target_owner == shooter and result.kind == f"base_{shooter}":
            rewards[shooter] -= 1.0
            infos[shooter]["own_base_blocked"] = True
            return
        if result.target_owner == shooter:
            rewards[shooter] -= 1.0
            infos[shooter]["own_target_blocked"] = result.target_name
            return
        target.knocked = True
        if result.kind == "normal":
            self.armor[opponent] = max(0, self.armor[opponent] - 1)
            self._fire_pose_cache.clear()
            self._path_cache.clear()
            self._route_distance_cache.clear()
            opponent_base_name = "BlueBaseTarget" if opponent == "blue" else "YellowBaseTarget"
            if self._normal_hits_against(shooter) >= int(self.base_retry_min_normal_hits.get(shooter, 0)):
                self.lost_targets[shooter].discard(opponent_base_name)
                self.target_cooldowns[shooter].pop(opponent_base_name, None)
            self.scores[shooter] += 5
            normal_hits_after = self._normal_hits_against(shooter)
            armor_break_bonus = (
                5.0
                if normal_hits_after == BASE_RUSH_PREFERRED_NORMAL_HITS
                else 3.2
                if normal_hits_after == BASE_RUSH_BALANCED_NORMAL_HITS
                else 1.8
                if normal_hits_after == BASE_RUSH_EARLY_NORMAL_HITS
                else 0.0
            )
            over_clear_penalty = 3.5 if normal_hits_after > BASE_RUSH_PREFERRED_NORMAL_HITS else 0.0
            rewards[shooter] += 10.0 + armor_break_bonus - over_clear_penalty
            rewards[opponent] -= 3.5
            self.strategy_counts[shooter]["normal_hits"] += 1
            self.target_order[shooter].append(target.name)
            self.post_hit_retreat_until[shooter] = self.elapsed + POST_HIT_RETREAT_S
            infos[shooter]["hit"] = result.target_name
            infos[shooter]["target_order"] = list(self.target_order[shooter])
            infos[shooter]["normal_hit_count"] = self.strategy_counts[shooter]["normal_hits"]
            return
        if result.kind == f"base_{opponent}":
            self.scores[shooter] += 60
            if terminal_override:
                self.winner = shooter
            self.strategy_counts[shooter]["base_hits"] += 1
            self.target_order[shooter].append(target.name)
            rewards[shooter] += 100.0
            rewards[opponent] -= 70.0
            if terminal_override:
                infos[shooter]["winner"] = shooter
            infos[shooter]["target_order"] = list(self.target_order[shooter])
