from __future__ import annotations

import math

import numpy as np

from .constants import (
    AGENTS,
    BASE_TARGET_CONTACT_RADIUS,
    NORMAL_TARGET_CONTACT_RADIUS,
    POST_HIT_RETREAT_S,
    PUSH_CLEARANCE_MARGIN,
    PUSH_ROBOT_RECOIL_M,
    PUSH_STEP_M
)
from .geometry import (
    robot_pushable_collision
)
from robocup_visionrl_gym_env import (
    BLUE_BASE_XY,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_RADIUS,
    YELLOW_BASE_XY
)


class ContactMixin:
    def _pushable_collision_name(self, pose: np.ndarray) -> str | None:
        for name, center in self.pushable_obstacles.items():
            collided, _normal, _penetration = robot_pushable_collision(
                pose,
                (float(center[0]), float(center[1])),
            )
            if collided:
                return name
        return None
    def _target_collision_name(self, pose: np.ndarray) -> str | None:
        x = float(pose[0])
        y = float(pose[1])
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            dx = target.xy[0] - x
            dy = target.xy[1] - y
            if dx * dx + dy * dy <= (ROBOT_RADIUS + radius) ** 2:
                return target.name
        return None
    def _push_obstacle(self, team: str, obstacle_name: str, robot_yaw: float, robot_xy: np.ndarray) -> bool:
        heading = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32)
        current = self.pushable_obstacles[obstacle_name]
        contact_normal = current - np.asarray(robot_xy, dtype=np.float32)
        contact_norm = float(np.linalg.norm(contact_normal))
        if contact_norm > 1e-6:
            contact_normal /= contact_norm
            if float(np.dot(heading, contact_normal)) < -0.10:
                return False
            direction = heading * 0.72 + contact_normal * 0.28
            direction_norm = float(np.linalg.norm(direction))
            direction = heading if direction_norm <= 1e-6 else direction / direction_norm
        else:
            direction = heading
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSH_CLEARANCE_MARGIN
        inflated = PUSHABLE_OBSTACLE_HALF + PUSH_CLEARANCE_MARGIN
        accepted = None
        for multiplier in (1.0, 1.7, 2.4, 3.1, 4.0):
            candidate = current + direction * (PUSH_STEP_M * self.domain_params.push_step_scale * multiplier)
            candidate = np.array(
                [
                    float(np.clip(candidate[0], -limit, limit)),
                    float(np.clip(candidate[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            robot_pose = np.array([float(robot_xy[0]), float(robot_xy[1]), float(robot_yaw)], dtype=np.float32)
            still_colliding, _normal, _penetration = robot_pushable_collision(
                robot_pose,
                (float(candidate[0]), float(candidate[1])),
                (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
            )
            if still_colliding:
                continue
            blocked = False
            for center, half_size in self.nav_blockers:
                if (
                    abs(float(candidate[0]) - center[0]) <= half_size[0] + inflated
                    and abs(float(candidate[1]) - center[1]) <= half_size[1] + inflated
                ):
                    blocked = True
                    break
            if blocked:
                continue
            for target in self.targets:
                if target.knocked:
                    continue
                target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
                if float(np.linalg.norm(candidate - np.asarray(target.xy, dtype=np.float32))) < inflated + target_radius:
                    blocked = True
                    break
            if blocked:
                continue
            for name, center in self.pushable_obstacles.items():
                if name == obstacle_name:
                    continue
                if float(np.linalg.norm(candidate - center)) < PUSHABLE_OBSTACLE_HALF * 2.0 + PUSH_CLEARANCE_MARGIN:
                    blocked = True
                    break
            if blocked:
                continue
            accepted = candidate
            break
        if accepted is None:
            return False
        self.pushable_obstacles[obstacle_name] = accepted
        self._fire_pose_cache.clear()
        self._path_cache.clear()
        self._route_distance_cache.clear()
        self.last_push_event[team] = obstacle_name
        self.last_push_impulse[team] = {
            "box": obstacle_name,
            "box_displacement_m": round(float(np.linalg.norm(accepted - current)), 4),
            "robot_recoil_m": PUSH_ROBOT_RECOIL_M,
        }
        return True
    def _apply_push_recoil_pose(self, pose: np.ndarray, motion_yaw: float, linear_speed: float) -> np.ndarray:
        recoil = PUSH_ROBOT_RECOIL_M * max(0.45, min(1.0, abs(float(linear_speed)) / 0.32))
        candidate = pose.copy()
        candidate[0] -= math.cos(float(motion_yaw)) * recoil
        candidate[1] -= math.sin(float(motion_yaw)) * recoil
        if self._static_pose_blocked(candidate) or self._target_collision_name(candidate) is not None:
            return pose
        return candidate
    def _separated_pose_from_pushable(self, pose: np.ndarray, obstacle_name: str) -> np.ndarray:
        center = self.pushable_obstacles[obstacle_name]
        collided, normal, penetration = robot_pushable_collision(
            pose,
            (float(center[0]), float(center[1])),
            (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
        )
        if not collided:
            return pose
        corrected = pose.copy()
        corrected[0] += normal[0] * (penetration + 0.008)
        corrected[1] += normal[1] * (penetration + 0.008)
        limit = HALF_ARENA - ROBOT_RADIUS - 0.012
        corrected[0] = float(np.clip(corrected[0], -limit, limit))
        corrected[1] = float(np.clip(corrected[1], -limit, limit))
        if self._static_pose_blocked(corrected) or self._target_collision_name(corrected) is not None:
            return pose
        return corrected
    def _separated_pose_from_all_pushables(self, pose: np.ndarray) -> np.ndarray:
        corrected = pose.copy()
        for _ in range(4):
            changed = False
            for _name, center in self.pushable_obstacles.items():
                collided, normal, penetration = robot_pushable_collision(
                    corrected,
                    (float(center[0]), float(center[1])),
                    (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
                )
                if not collided:
                    continue
                candidate = corrected.copy()
                candidate[0] += normal[0] * (penetration + 0.012)
                candidate[1] += normal[1] * (penetration + 0.012)
                limit = HALF_ARENA - ROBOT_RADIUS - 0.012
                candidate[0] = float(np.clip(candidate[0], -limit, limit))
                candidate[1] = float(np.clip(candidate[1], -limit, limit))
                if self._static_pose_blocked(candidate) or self._target_collision_name(candidate) is not None:
                    continue
                corrected = candidate
                changed = True
            if not changed:
                break
        return corrected
    def _resolve_contact(self) -> bool:
        delta = self.poses["blue"][:2] - self.poses["yellow"][:2]
        distance = float(np.linalg.norm(delta))
        min_distance = ROBOT_RADIUS * 2.0
        if distance >= min_distance:
            self.last_contact = False
            return False
        normal = np.array([1.0, 0.0], dtype=np.float32) if distance < 1e-6 else delta / distance
        push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
        before = {team: self.poses[team].copy() for team in AGENTS}
        yellow_candidate = before["yellow"].copy()
        blue_candidate = before["blue"].copy()
        yellow_candidate[:2] -= normal * push
        blue_candidate[:2] += normal * push
        self.poses["yellow"] = self._safe_contact_separation_pose(before["yellow"], yellow_candidate)
        self.poses["blue"] = self._safe_contact_separation_pose(before["blue"], blue_candidate)
        self.last_contact = True
        return True
    def _safe_contact_separation_pose(self, before: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        if not self._pose_blocked(candidate):
            return candidate
        repaired = candidate.copy()
        repaired[:2] = self._nearest_free_xy(candidate[:2])
        if not self._pose_blocked(repaired):
            return repaired
        return before
    def _contact_reward(self, team: str, info: dict[str, object]) -> float:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        opponent_distance_to_own_base = float(np.linalg.norm(self.poses[opponent][:2] - own_base))
        own_distance_to_own_base = float(np.linalg.norm(self.poses[team][:2] - own_base))
        tactical_intent = info.get("tactic") == "block" or bool(info.get("interference"))
        if tactical_intent and not self._near_own_critical_assets(team):
            threat = self._opponent_threat(team)
            return 0.09 + 0.16 * threat
        if opponent_distance_to_own_base < 0.85 and own_distance_to_own_base > 0.45:
            return 0.08
        if own_distance_to_own_base < 0.55:
            return -0.55
        return -0.035
    def _near_own_critical_assets(self, team: str) -> bool:
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        if float(np.linalg.norm(self.poses[team][:2] - own_base)) < 0.72:
            return True
        for target in self.targets:
            if target.owner != team or target.knocked:
                continue
            critical_radius = 0.34 if target.kind.startswith("base_") else 0.24
            if float(np.linalg.norm(self.poses[team][:2] - np.asarray(target.xy, dtype=np.float32))) < critical_radius:
                return True
        return False
    def _resolve_target_contacts(
        self,
        team: str,
        rewards: dict[str, float],
        infos: dict[str, dict[str, object]],
    ):
        if self.winner is not None:
            return
        pose_xy = self.poses[team][:2]
        for target in self.targets:
            if target.knocked:
                continue
            target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            distance = float(np.linalg.norm(np.array(target.xy, dtype=np.float32) - pose_xy))
            if distance > ROBOT_RADIUS + target_radius:
                continue
            infos[team]["target_collision"] = target.name
            if self.elapsed - self.last_target_contact_time[team] < 0.75:
                return
            self.last_target_contact_time[team] = self.elapsed
            self.post_hit_retreat_until[team] = self.elapsed + POST_HIT_RETREAT_S
            self._mark_target_failed(team, target.name)
            rewards[team] -= 1.2
            self.localization_confidence[team] = max(0.05, self.localization_confidence[team] - 0.03)
            if target.kind == f"base_{team}":
                infos[team]["own_base_collision"] = True
                rewards[team] -= 8.0
                return
            if target.kind.startswith("base_"):
                rewards[team] -= 3.0
                return
            rewards[team] -= 1.5
