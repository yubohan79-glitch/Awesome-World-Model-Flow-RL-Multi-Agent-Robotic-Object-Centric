from __future__ import annotations

import math

import numpy as np

from .constants import (
    BASE_AIM_MICRO_SCAN_RAD,
    BASE_AIM_SEEK_SCAN_RAD,
    BASE_IDEAL_CENTER_STANDOFF,
    BASE_RUSH_BALANCED_NORMAL_HITS,
    BASE_RUSH_EARLY_NORMAL_HITS,
    BASE_SHOT_CLOSE_DISTANCE,
    BASE_TARGET_CONTACT_RADIUS,
    FIRE_YAW_TOLERANCE_RAD,
    IDEAL_CENTER_STANDOFF,
    NORMAL_AIM_MICRO_SCAN_RAD,
    NORMAL_TARGET_CONTACT_RADIUS,
    SHOT_CLOSE_DISTANCE,
    TACTICAL_STANDOFF_MAX,
    TACTICAL_STANDOFF_MIN
)
from .geometry import (
    laser_origin_from_pose
)
from robocup_visionrl_gym_env import (
    BASE_HIT_RADIUS,
    BASE_SHOOT_RANGE,
    BLUE_BASE_XY,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_RADIUS,
    SHOOTER_FORWARD_OFFSET,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    Target,
    YELLOW_BASE_XY,
    active_base_armor_blockers,
    base_attack_pose_quality,
    base_hit_success_cap,
    shooting_range_limits,
    wrap_angle
)


class FireMixin:
    def _fire_standoff_goal(self, team: str, target: Target, risk: float) -> np.ndarray:
        if self._geometry_fire_ready(team, target, risk):
            return self.poses[team][:2].copy()
        solution = self._best_fire_pose(team, target, risk, route_aware=True)
        if solution is not None:
            return solution[0]
        target_xy = np.asarray(target.xy, dtype=np.float32)
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        return self._nearest_free_xy(target_xy + front * IDEAL_CENTER_STANDOFF)
    def _candidate_fire_poses(self, team: str, target: Target, risk: float) -> list[np.ndarray]:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        if target.kind.startswith("base_"):
            return self._candidate_base_fire_poses(team, target, risk)
        standoff = TACTICAL_STANDOFF_MAX - (TACTICAL_STANDOFF_MAX - TACTICAL_STANDOFF_MIN) * risk
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        tangent = np.array([-front[1], front[0]], dtype=np.float32)
        candidates: list[np.ndarray] = []
        for distance in (
            SHOT_CLOSE_DISTANCE,
            SHOOTER_FORWARD_OFFSET + 0.24,
            IDEAL_CENTER_STANDOFF,
            standoff,
            SHOOTER_FORWARD_OFFSET + 0.42,
            SHOOTER_FORWARD_OFFSET + SHOOT_RANGE - 0.03,
        ):
            for lateral in (0.0, -0.08, 0.08):
                candidates.append(target_xy + front * distance + tangent * lateral)
        return candidates
    def _candidate_base_fire_poses(self, team: str, target: Target, risk: float) -> list[np.ndarray]:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        front = np.array([math.cos(target.yaw), math.sin(target.yaw)], dtype=np.float32)
        tangent = np.array([-front[1], front[0]], dtype=np.float32)
        wall_limit = HALF_ARENA - ROBOT_RADIUS - 0.045
        edge_offsets = (-0.16, -0.08, 0.0, 0.08, 0.16)
        candidates: list[np.ndarray] = []
        hits = max(1, min(4, self._normal_hits_against(team)))
        base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
        if target.kind == "base_blue":
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
        allowed_dirs = opened_dirs[:1] if hits == 1 else [opened_dirs[1], opened_dirs[0]] if hits == 2 else opened_dirs
        side_radii = (0.62,) if hits == 1 else (0.48, 0.62, 0.78) if hits == 2 else (0.48, 0.62, 0.78, 0.94)
        side_laterals = (0.0,) if hits == 1 else (-0.06, 0.0, 0.06) if hits == 2 else (-0.10, -0.04, 0.0, 0.04, 0.10)
        fine_radial_offsets = (0.0, -0.055, -0.035, -0.018, 0.018, 0.035, 0.055) if hits >= 2 else (0.0,)
        fine_lateral_offsets = (0.0, -0.055, -0.035, -0.015, 0.015, 0.035, 0.055) if hits >= 2 else (0.0,)
        for direction in allowed_dirs:
            side_tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
            for radius in side_radii:
                for lateral in side_laterals:
                    anchor = np.asarray(base_xy, dtype=np.float32) + direction * radius + side_tangent * lateral
                    candidates.append(anchor)
                    for radial_delta in fine_radial_offsets:
                        for lateral_delta in fine_lateral_offsets:
                            if abs(radial_delta) <= 1e-6 and abs(lateral_delta) <= 1e-6:
                                continue
                            candidates.append(anchor + direction * radial_delta + side_tangent * lateral_delta)
        if hits >= 2:
            if target.kind == "base_blue":
                for offset in edge_offsets:
                    candidates.append(np.array([-wall_limit, float(target_xy[1] + offset)], dtype=np.float32))
                    candidates.append(np.array([float(target_xy[0] + offset), wall_limit], dtype=np.float32))
            else:
                for offset in edge_offsets:
                    candidates.append(np.array([wall_limit, float(target_xy[1] + offset)], dtype=np.float32))
                    candidates.append(np.array([float(target_xy[0] + offset), -wall_limit], dtype=np.float32))
        # Add diagonal center-facing options for late base attacks after more
        # armor has been removed; the validation step still rejects them during
        # one- or two-target rushes.
        if hits >= 2:
            angle_offsets = {
                2: (-0.96, 0.96, -0.70, 0.70),
                3: (-0.78, 0.78, -0.44, 0.44, -0.26, 0.26),
                4: (-0.62, 0.62, -0.34, 0.34, -0.16, 0.16, 0.0),
            }[hits]
            for distance in (
                BASE_SHOT_CLOSE_DISTANCE,
                SHOOTER_FORWARD_OFFSET + 0.34,
                BASE_IDEAL_CENTER_STANDOFF,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_RANGE - 0.04,
            ):
                for offset in angle_offsets:
                    direction = np.array(
                        [
                            math.cos(float(target.yaw) + offset),
                            math.sin(float(target.yaw) + offset),
                        ],
                        dtype=np.float32,
                    )
                    candidates.append(target_xy + direction * distance)
            for distance in (
                BASE_SHOT_CLOSE_DISTANCE,
                SHOOTER_FORWARD_OFFSET + 0.36,
                BASE_IDEAL_CENTER_STANDOFF,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_RANGE - 0.03,
            ):
                for lateral in (0.0, -0.04, 0.04, -0.10, 0.10, -0.18, 0.18):
                    candidates.append(target_xy + front * distance + tangent * lateral)
        return candidates
    def _laser_origin_for_fire_pose(self, candidate: np.ndarray, target_xy: np.ndarray) -> tuple[float, float]:
        delta = target_xy - candidate
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return (float(candidate[0]), float(candidate[1]))
        forward = delta / distance
        origin = candidate + forward * SHOOTER_FORWARD_OFFSET
        return (float(origin[0]), float(origin[1]))
    def _base_line_clear_with_yaw_margin(self, team: str, candidate: np.ndarray, target: Target) -> bool:
        target_xy = np.asarray(target.xy, dtype=np.float32)
        base_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
        for yaw_delta in (-0.035, 0.0, 0.035):
            yaw = base_yaw + yaw_delta
            origin = (
                float(candidate[0]) + SHOOTER_FORWARD_OFFSET * math.cos(yaw),
                float(candidate[1]) + SHOOTER_FORWARD_OFFSET * math.sin(yaw),
            )
            if not self._target_line_clear(team, origin, target):
                return False
        return True
    def _valid_fire_pose_candidates(
        self,
        team: str,
        target: Target,
        risk: float,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        if target.kind.startswith("base_"):
            normal_hits = self._normal_hits_against(team)
            if normal_hits < BASE_RUSH_EARLY_NORMAL_HITS:
                return []
            if normal_hits < int(self.base_retry_min_normal_hits.get(team, 0)):
                if normal_hits < BASE_RUSH_BALANCED_NORMAL_HITS or self._has_available_normal_retry_target(team):
                    return []
        risk_bucket = int(round(float(np.clip(risk, 0.0, 1.0)) * 4.0))
        cache_key = (target.name, risk_bucket, int(self.armor[target.owner]))
        cached = self._fire_pose_cache.get(cache_key)
        if cached is not None:
            return cached
        bucket_risk = risk_bucket / 4.0
        target_xy = np.asarray(target.xy, dtype=np.float32)
        valid: list[tuple[np.ndarray, float, float, float]] = []
        seen: set[tuple[int, int]] = set()
        for raw_candidate in self._candidate_fire_poses(team, target, bucket_risk):
            candidate = np.asarray(raw_candidate, dtype=np.float32)
            firing_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
            pose_candidate = np.array([candidate[0], candidate[1], firing_yaw], dtype=np.float32)
            if self._pose_blocked(pose_candidate):
                candidate = self._nearest_free_xy(candidate)
                firing_yaw = math.atan2(float(target_xy[1] - candidate[1]), float(target_xy[0] - candidate[0]))
                pose_candidate = np.array([candidate[0], candidate[1], firing_yaw], dtype=np.float32)
            key = (round(float(candidate[0]) * 100), round(float(candidate[1]) * 100))
            if key in seen:
                continue
            seen.add(key)
            if self._pose_blocked(pose_candidate):
                continue
            if target.kind.startswith("base_") and self._base_attack_pose_quality(team, target, candidate) <= 0.0:
                continue
            laser_origin = self._laser_origin_for_fire_pose(candidate, target_xy)
            if not self._target_line_clear(team, laser_origin, target):
                continue
            if target.kind.startswith("base_") and not self._base_line_clear_with_yaw_margin(team, candidate, target):
                continue
            center_distance = float(np.linalg.norm(candidate - target_xy))
            shot_distance = max(0.0, center_distance - SHOOTER_FORWARD_OFFSET)
            contact_min = self._target_contact_clearance(target)
            min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
            if center_distance < contact_min or shot_distance < min_range or shot_distance > max_range:
                continue
            shot_quality = self._shot_accuracy_from_geometry(shot_distance, 0.0, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                shot_quality *= self._base_attack_pose_quality(team, target, candidate)
            blocker_cost = self._local_blocker_cost(candidate)
            valid.append((candidate, shot_distance, shot_quality, blocker_cost))
        self._fire_pose_cache[cache_key] = valid
        return valid
    def _fire_geometry_snapshot(self, team: str, target: Target, risk: float) -> dict[str, object]:
        pose = self.poses[team]
        center_dx = target.xy[0] - float(pose[0])
        center_dy = target.xy[1] - float(pose[1])
        center_distance = math.hypot(center_dx, center_dy)
        origin = laser_origin_from_pose(pose)
        dx = target.xy[0] - origin[0]
        dy = target.xy[1] - origin[1]
        forward = (math.cos(float(pose[2])), math.sin(float(pose[2])))
        distance = dx * forward[0] + dy * forward[1]
        lateral_error = abs(dx * forward[1] - dy * forward[0])
        bearing = math.atan2(dy, dx)
        yaw_error = abs(wrap_angle(bearing - float(pose[2])))
        angle_threshold = FIRE_YAW_TOLERANCE_RAD + 0.035 * risk
        if target.kind.startswith("base_"):
            angle_threshold = min(0.24 + 0.04 * risk, 0.17 + 0.04 * self._normal_hits_against(team))
        line_clear = self._target_line_clear(team, origin, target)
        base_pose_ok = True
        if target.kind.startswith("base_"):
            base_pose_ok = self._base_attack_pose_quality(team, target, pose[:2]) > 0.0
        hit_radius = BASE_HIT_RADIUS if target.kind.startswith("base_") else SHOOT_HIT_RADIUS
        min_range, max_range = shooting_range_limits(target.kind.startswith("base_"))
        geometry_ready = bool(
            target.owner != team
            and yaw_error < angle_threshold
            and lateral_error <= hit_radius
            and center_distance >= self._target_contact_clearance(target)
            and distance >= min_range
            and distance <= max_range
            and line_clear
            and base_pose_ok
        )
        return {
            "center_distance": center_distance,
            "shot_distance": distance,
            "lateral_error": lateral_error,
            "yaw_error": yaw_error,
            "angle_threshold": angle_threshold,
            "hit_radius": hit_radius,
            "line_clear": line_clear,
            "base_pose_ok": base_pose_ok,
            "geometry_ready": geometry_ready,
        }
    def _geometry_fire_ready(self, team: str, target: Target, risk: float) -> bool:
        return bool(self._fire_geometry_snapshot(team, target, risk)["geometry_ready"])
    def _hold_fire_pose(self, team: str, target: Target, risk: float) -> bool:
        pose = self.poses[team]
        desired_yaw = math.atan2(target.xy[1] - float(pose[1]), target.xy[0] - float(pose[0]))
        geometry = self._fire_geometry_snapshot(team, target, risk)
        base_target = target.kind.startswith("base_")
        if bool(geometry["geometry_ready"]):
            shot_distance = max(float(geometry["shot_distance"]), 1e-6)
            hit_radius = float(geometry["hit_radius"])
            lateral_error = float(geometry["lateral_error"])
            margin_rad = max(0.0, hit_radius - lateral_error) / shot_distance
            max_scan = BASE_AIM_MICRO_SCAN_RAD if base_target else NORMAL_AIM_MICRO_SCAN_RAD
            scan_amp = min(max_scan, 0.45 * margin_rad)
            if scan_amp > 0.002:
                phase_offset = 0.0 if team == "yellow" else math.pi * 0.5
                frequency_hz = 0.42 if base_target else 0.58
                desired_yaw = wrap_angle(
                    desired_yaw + scan_amp * math.sin(math.tau * frequency_hz * self.elapsed + phase_offset)
                )
        elif base_target and bool(geometry["line_clear"]) and bool(geometry["base_pose_ok"]):
            min_range, max_range = shooting_range_limits(True)
            shot_distance = max(float(geometry["shot_distance"]), 1e-6)
            if min_range <= shot_distance <= max_range:
                hit_radius = float(geometry["hit_radius"])
                lateral_error = float(geometry["lateral_error"])
                # At a legal base fire pose a centimeter of pose error can leave
                # the laser just outside the small base hit radius. Keep a slow
                # deterministic search alive instead of freezing at a single yaw.
                seek_amp = min(BASE_AIM_SEEK_SCAN_RAD, max(BASE_AIM_MICRO_SCAN_RAD, 0.30 * lateral_error / shot_distance))
                if lateral_error > 0.50 * hit_radius and seek_amp > 0.002:
                    phase_offset = math.pi * 0.25 if team == "yellow" else math.pi * 0.75
                    desired_yaw = wrap_angle(
                        desired_yaw + seek_amp * math.sin(math.tau * 0.30 * self.elapsed + phase_offset)
                    )
        yaw_error = wrap_angle(desired_yaw - float(pose[2]))
        settling_deadband = 0.004 if bool(geometry["geometry_ready"]) else 0.012
        angular_speed = 0.0 if abs(yaw_error) < settling_deadband else float(np.clip(2.25 * yaw_error, -0.72, 0.72))
        return self._integrate_command(team, 0.0, angular_speed, allow_push=False)
    def _target_line_clear(self, team: str, origin: tuple[float, float], target: Target) -> bool:
        return not self._line_blocked(origin, target.xy)
    def _center_aim_line_clear(self, team: str, target: Target) -> bool:
        pose = self.poses[team].copy()
        pose[2] = math.atan2(target.xy[1] - float(pose[1]), target.xy[0] - float(pose[0]))
        return self._target_line_clear(team, laser_origin_from_pose(pose), target)
    def _best_fire_pose(
        self,
        team: str,
        target: Target,
        risk: float,
        *,
        route_aware: bool = True,
    ) -> tuple[np.ndarray, float, float, float] | None:
        pose = self.poses[team]
        scored: list[tuple[float, np.ndarray, float, float, float]] = []
        blocked_candidates: list[tuple[float, np.ndarray, float, float, float]] = []
        blocker_weight = 0.42 if target.kind.startswith("base_") else 0.20
        for candidate, shot_distance, shot_quality, blocker_cost in self._valid_fire_pose_candidates(team, target, risk):
            direct_distance = float(np.linalg.norm(candidate - pose[:2]))
            direct_blocked = self._segment_likely_blocked_for_nav(pose[:2], candidate)
            if not route_aware:
                route_distance = direct_distance + (1.65 if direct_blocked else 0.0)
                score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
                scored.append((score, candidate, route_distance, shot_distance, shot_quality))
                continue
            if direct_blocked:
                estimated_distance = direct_distance + 0.72 + 0.20 * blocker_cost
                quick_score = shot_quality - 0.16 * estimated_distance - blocker_weight * blocker_cost
                blocked_candidates.append((quick_score, candidate, estimated_distance, shot_distance, shot_quality))
                continue
            route_distance = direct_distance
            score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
            scored.append((score, candidate, route_distance, shot_distance, shot_quality))
        if route_aware and blocked_candidates:
            best_direct = max((item[0] for item in scored), default=-math.inf)
            best_blocked = max(item[0] for item in blocked_candidates)
            if not scored or best_blocked > best_direct - 0.06:
                blocked_candidates.sort(key=lambda item: item[0], reverse=True)
                for _quick_score, candidate, _estimated_distance, shot_distance, shot_quality in blocked_candidates[:6]:
                    route_distance = self._route_distance_to(pose[:2], candidate)
                    if not math.isfinite(route_distance):
                        continue
                    blocker_cost = self._local_blocker_cost(candidate)
                    score = shot_quality - 0.16 * route_distance - blocker_weight * blocker_cost
                    scored.append((score, candidate, route_distance, shot_distance, shot_quality))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            _score, fire_xy, route_distance, shot_distance, shot_quality = scored[0]
            return fire_xy, route_distance, shot_distance, shot_quality
        return None
    def _update_fire_gate(self, team: str, target: Target, action: np.ndarray, risk: float) -> dict[str, object]:
        geometry = self._fire_geometry_snapshot(team, target, risk)
        center_distance = float(geometry["center_distance"])
        distance = float(geometry["shot_distance"])
        lateral_error = float(geometry["lateral_error"])
        yaw_error = float(geometry["yaw_error"])
        angle_threshold = float(geometry["angle_threshold"])
        line_clear = bool(geometry["line_clear"])
        fire_gate = float(action[4])
        action_shield_fire = bool(self.action_shield and fire_gate > -0.25 and not bool(geometry["geometry_ready"]))
        if action_shield_fire:
            fire_gate = -1.0
        self.pending_fire[team] = (
            fire_gate > -0.25
            and bool(geometry["geometry_ready"])
        )
        if not self.pending_fire[team]:
            self._reset_laser_lock(team)
        return {
            "shot_distance_m": round(float(distance), 4),
            "center_target_distance_m": round(float(center_distance), 4),
            "shot_yaw_error_rad": round(float(yaw_error), 4),
            "shot_lateral_error_m": round(float(lateral_error), 4),
            "shot_accuracy_estimate": round(
                self._shot_accuracy_from_geometry(
                    distance,
                    lateral_error,
                    target.kind.startswith("base_"),
                ),
                4,
            ),
            "line_clear": line_clear,
            "fire_yaw_threshold_rad": round(float(angle_threshold), 4),
            "action_shield_fire": action_shield_fire,
        }
    def _target_contact_clearance(self, target: Target) -> float:
        target_radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
        return ROBOT_RADIUS + target_radius + 0.004
    def _base_attack_pose_quality(self, team: str, target: Target, xy: np.ndarray) -> float:
        if not target.kind.startswith("base_"):
            return 1.0
        hits = self._normal_hits_against(team)
        if hits < BASE_RUSH_EARLY_NORMAL_HITS:
            return 0.0
        base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
        return base_attack_pose_quality(hits, target.xy, target.yaw, base_xy, xy)
    def _base_hit_cap_for_team(self, team: str) -> float:
        return base_hit_success_cap(self._normal_hits_against(team))
    def _shot_accuracy_from_geometry(self, distance: float, lateral_error: float, base_target: bool) -> float:
        min_range, max_range = shooting_range_limits(base_target)
        if distance < min_range or distance > max_range:
            return 0.0
        max_lateral = BASE_HIT_RADIUS if base_target else SHOOT_HIT_RADIUS
        if lateral_error > max_lateral:
            return 0.0
        distance_quality = (max_range - distance) / max(1e-6, max_range - min_range)
        lateral_quality = 1.0 - lateral_error / max(max_lateral, 1e-6)
        # Close, centered shots are reliable; far-edge shots are intentionally
        # uncertain so the policy learns the time-vs-accuracy tradeoff.
        accuracy = 0.18 + 0.64 * distance_quality + 0.18 * lateral_quality
        if base_target:
            accuracy -= 0.10
        return float(np.clip(accuracy, 0.05, 0.98))
    def _local_blocker_cost(self, point: np.ndarray) -> float:
        x, y = float(point[0]), float(point[1])
        cost = 0.0
        for center, half_size in self.nav_blockers:
            dx = max(0.0, abs(x - center[0]) - half_size[0])
            dy = max(0.0, abs(y - center[1]) - half_size[1])
            distance = math.hypot(dx, dy)
            cost += max(0.0, 0.18 - distance) / 0.18
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            dx = max(0.0, abs(x - center[0]) - half_size[0])
            dy = max(0.0, abs(y - center[1]) - half_size[1])
            distance = math.hypot(dx, dy)
            cost += 1.15 * max(0.0, 0.20 - distance) / 0.20
        for center in self.pushable_obstacles.values():
            dx = max(0.0, abs(x - float(center[0])) - PUSHABLE_OBSTACLE_HALF)
            dy = max(0.0, abs(y - float(center[1])) - PUSHABLE_OBSTACLE_HALF)
            distance = math.hypot(dx, dy)
            cost += 0.55 * max(0.0, 0.18 - distance) / 0.18
        return cost
