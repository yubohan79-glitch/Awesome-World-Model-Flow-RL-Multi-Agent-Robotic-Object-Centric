from __future__ import annotations

import math

import numpy as np

from .constants import (
    AGENTS,
    BASE_RUSH_BALANCED_NORMAL_HITS,
    BASE_RUSH_EARLY_NORMAL_HITS,
    BASE_RUSH_PREFERRED_NORMAL_HITS,
    CAMERA_MEMORY_FOV_RAD,
    CAMERA_MEMORY_RANGE_M,
    IDEAL_SHOOT_DISTANCE
)
from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BLUE_BASE_XY,
    Target,
    YELLOW_BASE_XY,
    wrap_angle
)


class TacticsMixin:
    def _select_tactical_target(self, team: str, action: np.ndarray) -> Target | None:
        opponent = self._opponent(team)
        self._refresh_target_visibility_memory(team)
        normal_targets = [
            target for target in self.targets
            if target.kind == "normal"
            and target.owner == opponent
            and not target.knocked
            and not self._target_on_cooldown(team, target.name)
        ]
        base_targets = [
            target for target in self.targets
            if target.kind == f"base_{opponent}"
            and not target.knocked
            and not self._target_on_cooldown(team, target.name)
        ]
        normal_hits = self._normal_hits_against(team)
        risk = (float(action[5]) + 1.0) * 0.5
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        low_time = time_remaining < 45.0
        early_base_commit = (
            normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS
            or (
                normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS
                and (float(action[1]) > 0.45 or low_time)
                and risk > 0.62
            )
            or (
                normal_hits == BASE_RUSH_EARLY_NORMAL_HITS
                and (float(action[1]) > 0.72 or low_time)
                and risk > 0.80
            )
        )
        if self._base_rush_open(team, action) and base_targets and float(action[1]) > -0.35 and early_base_commit:
            reachable_base_targets = [
                target for target in base_targets
                if self._best_fire_pose(team, target, (float(action[5]) + 1.0) * 0.5, route_aware=True) is not None
            ]
            if reachable_base_targets:
                return reachable_base_targets[0]
        base_gate = float(action[1])
        allow_base = (
            (
                self._normal_hits_against(team) >= BASE_RUSH_BALANCED_NORMAL_HITS
                and base_gate > 0.22
                and risk > 0.50
            )
            or self._base_rush_open(team, action)
            or low_time
        )
        candidates = list(normal_targets)
        if allow_base or not candidates:
            candidates.extend(base_targets)
        if not candidates:
            return None
        scored_candidates = [
            (self._target_priority(team, target, action, risk, route_aware=True), target)
            for target in candidates
        ]
        ranked = [
            target for priority, target in sorted(scored_candidates, key=lambda item: item[0], reverse=True)
            if priority > -50.0
        ]
        if not ranked:
            return None
        if len(ranked) == 1:
            return ranked[0]
        selector = (float(action[0]) + 1.0) * 0.5
        near_window = min(3, len(ranked))
        index = int(round(selector * (near_window - 1)))
        return ranked[max(0, min(near_window - 1, index))]
    def _target_on_cooldown(self, team: str, target_name: str) -> bool:
        if target_name.endswith("BaseTarget"):
            min_hits = int(self.base_retry_min_normal_hits.get(team, 0))
            if self._normal_hits_against(team) < min_hits:
                if (
                    self._normal_hits_against(team) >= BASE_RUSH_BALANCED_NORMAL_HITS
                    and not self._has_available_normal_retry_target(team)
                ):
                    return False
                return True
            self.lost_targets[team].discard(target_name)
            if self.target_cooldowns[team].get(target_name, -99.0) > self.max_time_s:
                self.target_cooldowns[team].pop(target_name, None)
        if target_name in self.lost_targets[team] and self.elapsed >= self.target_cooldowns[team].get(target_name, -99.0):
            self.lost_targets[team].discard(target_name)
        return target_name in self.lost_targets[team] or self.elapsed < self.target_cooldowns[team].get(target_name, -99.0)
    def _has_available_normal_retry_target(self, team: str) -> bool:
        opponent = self._opponent(team)
        for target in self.targets:
            if target.kind != "normal" or target.owner != opponent or target.knocked:
                continue
            if target.name in self.lost_targets[team] and self.elapsed >= self.target_cooldowns[team].get(target.name, -99.0):
                self.lost_targets[team].discard(target.name)
            if target.name in self.lost_targets[team] or self.elapsed < self.target_cooldowns[team].get(target.name, -99.0):
                continue
            if self._best_fire_pose(team, target, risk=0.82, route_aware=True) is not None:
                return True
        return False
    def _mark_target_failed(self, team: str, target_name: str):
        count = self.target_fail_counts[team].get(target_name, 0) + 1
        self.target_fail_counts[team][target_name] = count
        target = next((item for item in self.targets if item.name == target_name), None)
        if target_name.endswith("BaseTarget") or (target is not None and target.kind.startswith("base_")):
            self.lost_targets[team].discard(target_name)
            self.target_cooldowns[team][target_name] = self.elapsed + 4.0 + 1.5 * min(count, 4)
            return
        if target is not None and not target.knocked and not self._target_visible_in_camera(team, target):
            self.lost_targets[team].add(target_name)
            self.target_cooldowns[team][target_name] = self.elapsed + 8.0 + 3.0 * min(count, 4)
            return
        self.target_cooldowns[team][target_name] = self.elapsed + 6.0 + 2.5 * min(count, 4)
    def _refresh_target_visibility_memory(self, team: str):
        opponent = self._opponent(team)
        for target in self.targets:
            if target.owner != opponent or target.kind != "normal" or target.knocked:
                continue
            if self.target_fail_counts[team].get(target.name, 0) <= 0:
                continue
            if not self._target_visible_in_camera(team, target):
                self.lost_targets[team].add(target.name)
                self.target_cooldowns[team][target.name] = max(
                    self.target_cooldowns[team].get(target.name, -99.0),
                    self.elapsed + 3.0,
                )
    def _target_visible_in_camera(self, team: str, target: Target) -> bool:
        pose = self.poses[team]
        delta = np.asarray(target.xy, dtype=np.float32) - pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance > CAMERA_MEMORY_RANGE_M or distance < 1e-6:
            return False
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        if abs(wrap_angle(bearing - float(pose[2]))) > CAMERA_MEMORY_FOV_RAD * 0.5:
            return False
        return not self._line_blocked((float(pose[0]), float(pose[1])), target.xy)
    def _normal_hits_against(self, team: str) -> int:
        opponent = self._opponent(team)
        return max(0, 4 - int(self.armor[opponent]))
    def _base_rush_open(self, team: str, action: np.ndarray | None = None) -> bool:
        hits = self._normal_hits_against(team)
        if hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
            return True
        if hits < BASE_RUSH_EARLY_NORMAL_HITS:
            return False
        if action is None:
            return False
        risk = (float(action[5]) + 1.0) * 0.5
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        if hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
            return (float(action[1]) > 0.42 and risk > 0.58) or time_remaining < 38.0
        return (float(action[1]) > 0.72 and risk > 0.80) or time_remaining < 30.0
    def _target_priority(
        self,
        team: str,
        target: Target,
        action: np.ndarray,
        risk: float,
        *,
        route_aware: bool,
    ) -> float:
        opponent = self._opponent(team)
        solution = self._best_fire_pose(team, target, risk, route_aware=route_aware)
        if solution is None:
            return -999.0
        _fire_xy, route_distance, shot_distance, shot_quality = solution
        priority = 1.35 - route_distance / (ARENA_SIZE * 0.90)
        priority += 0.65 * shot_quality
        priority += 0.18 * max(0.0, 1.0 - abs(shot_distance - IDEAL_SHOOT_DISTANCE) / 0.22)
        normal_hits = self._normal_hits_against(team)
        if target.kind == "normal":
            priority += 0.42 + 0.14 * min(normal_hits, BASE_RUSH_BALANCED_NORMAL_HITS)
            if normal_hits == 0 and target.name in {"T03_WestAboveGate", "T06_EastBelowGate"}:
                priority += 0.64
            if normal_hits == 1 and target.name in {"T03_WestAboveGate", "T06_EastBelowGate"}:
                priority += 0.46
            if normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                priority -= 0.28
            if normal_hits >= 4:
                priority -= 1.15
        if target.kind == f"base_{opponent}":
            cap = self._base_hit_cap_for_team(team)
            priority += 0.42 * risk + 0.40 * float(action[1]) + 0.82 * cap
            if normal_hits == BASE_RUSH_BALANCED_NORMAL_HITS:
                priority_team = getattr(self, "base_rush_priority_team", None)
                if priority_team in AGENTS:
                    priority += 0.36 if team == priority_team else -0.58
            if normal_hits >= 4:
                priority += 1.08
            elif normal_hits >= BASE_RUSH_PREFERRED_NORMAL_HITS:
                priority += 0.72
            elif normal_hits >= BASE_RUSH_BALANCED_NORMAL_HITS:
                priority -= 0.12
            elif normal_hits >= BASE_RUSH_EARLY_NORMAL_HITS:
                priority -= 0.82
            else:
                priority -= 1.20
        return priority
    def _opponent_threat(self, team: str) -> float:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        other = self.poses[opponent]
        base_delta = own_base - other[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(other[2])
        heading_error = abs(wrap_angle(base_bearing - float(other[2])))
        proximity = max(0.0, 1.0 - base_distance / 1.10)
        heading = max(0.0, 1.0 - heading_error / math.pi)
        visible = not self._line_blocked((float(self.poses[team][0]), float(self.poses[team][1])), (float(other[0]), float(other[1])))
        return max(0.0, min(1.0, proximity * (0.55 + 0.45 * heading) * (1.0 if visible else 0.72)))
    def _nearest_opponent_target_distance(self, team: str) -> float:
        opponent = self._opponent(team)
        pose = self.poses[team]
        candidates = [
            target for target in self.targets
            if target.owner == opponent and not target.knocked
        ]
        if not candidates:
            return 0.0
        return min(float(np.linalg.norm(np.asarray(target.xy, dtype=np.float32) - pose[:2])) for target in candidates)
