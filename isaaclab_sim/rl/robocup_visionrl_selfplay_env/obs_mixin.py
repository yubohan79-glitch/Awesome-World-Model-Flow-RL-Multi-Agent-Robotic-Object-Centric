from __future__ import annotations

import math

import numpy as np

from .constants import (
    ACCEL_DRIFT_ANGULAR_THRESHOLD,
    ACCEL_DRIFT_LINEAR_THRESHOLD,
    ACCEL_DRIFT_LOSS_SCALE,
    FUSION_CONFIDENCE_DRIFT_LOSS,
    FUSION_CONFIDENCE_RECOVERY_GAIN,
    FUSION_HARD_CONTACT_LOSS,
    FUSION_JAMMED_PUSH_LOSS,
    RECOVERY_COOLDOWN_S,
    TOF_SENSOR_LATERAL_OFFSET_M,
    TOF_SENSOR_RANGE_M
)
from .geometry import (
    team_frame_sign
)
from robocup_visionrl_gym_env import (
    ARENA_SIZE,
    BLUE_BASE_XY,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_LENGTH,
    Target,
    YELLOW_BASE_XY,
    active_base_armor_blockers,
    wrap_angle
)


class ObsMixin:
    def _opponent_tracking_features(self, team: str, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        delta = other[:2] - own[:2]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else float(own[2])
        relative_bearing = wrap_angle(bearing - float(own[2]))
        visible = 0.0 if self._line_blocked((float(own[0]), float(own[1])), (float(other[0]), float(other[1]))) else 1.0
        base_delta = own_base - other[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(other[2])
        heading_to_own_base = abs(wrap_angle(base_bearing - float(other[2])))
        proximity_threat = max(0.0, 1.0 - base_distance / 1.10)
        heading_threat = max(0.0, 1.0 - heading_to_own_base / math.pi)
        threat = max(0.0, min(1.0, proximity_threat * (0.55 + 0.45 * heading_threat) * (1.0 if visible else 0.72)))
        return np.array(
            [
                distance / ARENA_SIZE,
                math.cos(relative_bearing),
                math.sin(relative_bearing),
                visible,
                threat,
            ],
            dtype=np.float32,
        )
    def _obs(self, team: str) -> np.ndarray:
        opponent = self._opponent(team)
        own = self.poses[team]
        other = self.poses[opponent]
        opponent_base = BLUE_BASE_XY if team == "yellow" else YELLOW_BASE_XY
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        normal_targets = [target for target in self.targets if target.kind == "normal"]
        active_opponent_normals = [
            target for target in normal_targets if not target.knocked and target.owner == opponent
        ]
        fire_solutions = [
            self._best_fire_pose(team, target, risk=0.35, route_aware=False)
            for target in active_opponent_normals
        ]
        reachable_fire_xy = [solution[0] for solution in fire_solutions if solution is not None]
        if reachable_fire_xy:
            nearest_fire_xy = min(reachable_fire_xy, key=lambda xy: np.linalg.norm(xy - own[:2]))
            nearest_vec = self._team_vector(team, nearest_fire_xy - own[:2]) / ARENA_SIZE
        else:
            nearest_vec = np.zeros(2, dtype=np.float32)
        knocked_flags = self._canonical_target_flags(team, normal_targets)
        rel_opponent = self._team_vector(team, other[:2] - own[:2]) / ARENA_SIZE
        opponent_track = self._opponent_tracking_features(team, own, other)
        pushable_items = sorted(
            self.pushable_obstacles.items(),
            key=lambda item: tuple(round(float(v), 4) for v in self._team_vector(team, item[1] - own[:2])),
        )
        pushable_vectors = np.concatenate(
            [
                self._team_vector(team, center - own[:2]) / ARENA_SIZE
                for _name, center in pushable_items
            ]
        ).astype(np.float32)
        own_xy = self._team_point(team, own[:2])
        own_yaw = self._team_yaw(team, float(own[2]))
        other_yaw = self._team_yaw(team, float(other[2]))
        obs = np.concatenate(
            [
                np.array([own_xy[0] / HALF_ARENA, own_xy[1] / HALF_ARENA, math.cos(own_yaw), math.sin(own_yaw)]),
                rel_opponent,
                opponent_track,
                np.array([math.cos(other_yaw), math.sin(other_yaw)]),
                np.array([self.armor[opponent] / 4.0, self.armor[team] / 4.0]),
                np.array([(self.scores[team] - self.scores[opponent]) / 60.0, self.elapsed / self.max_time_s]),
                self._sensor_fusion_features(team),
                knocked_flags,
                nearest_vec,
                pushable_vectors,
                self._team_vector(team, opponent_base - own[:2]) / ARENA_SIZE,
                self._team_vector(team, own_base - own[:2]) / ARENA_SIZE,
                np.array([1.0 if self.winner == team else -1.0 if self.winner == opponent else 0.0]),
                np.array([1.0 if team == "yellow" else -1.0]),
            ]
        ).astype(np.float32)
        return obs
    def _default_sensor_fusion_state(self) -> dict[str, float]:
        return {
            "wheel_imu_consistency": 1.0,
            "scan_clearance": 1.0,
            "tof_front_left_clearance": 1.0,
            "tof_front_right_clearance": 1.0,
            "bumper_or_hard_contact": 0.0,
            "camera_target_visible": 1.0,
            "pushable_contact": 0.0,
        }
    def _sensor_fusion_features(self, team: str) -> np.ndarray:
        fusion = self.sensor_fusion[team]
        if self.domain_params.sensor_noise_scale > 0.0:
            noise = self.domain_params.sensor_noise_scale
            fusion = {
                key: (
                    float(np.clip(value + self.rng.normal(0.0, noise), 0.0, 1.0))
                    if key not in ("bumper_or_hard_contact", "pushable_contact")
                    else value
                )
                for key, value in fusion.items()
            }
        return np.array(
            [
                float(self.last_contact),
                float(self.localization_confidence[team]),
                float(fusion["wheel_imu_consistency"]),
                float(fusion["scan_clearance"]),
                float(fusion["tof_front_left_clearance"]),
                float(fusion["tof_front_right_clearance"]),
                float(fusion["bumper_or_hard_contact"]),
                float(fusion["camera_target_visible"]),
                float(fusion["pushable_contact"]),
            ],
            dtype=np.float32,
        )
    def _record_motion_sensor_fusion(
        self,
        team: str,
        before: np.ndarray,
        after: np.ndarray,
        linear_speed: float,
        angular_speed: float,
        *,
        blocked: bool,
        hard_contact: bool = False,
        push_contact: bool = False,
        jammed_push: bool = False,
    ):
        expected_distance = abs(float(linear_speed)) * self.dt
        actual_distance = float(np.linalg.norm(after[:2] - before[:2]))
        expected_yaw = abs(float(angular_speed)) * self.dt
        actual_yaw = abs(wrap_angle(float(after[2] - before[2])))
        translation_error = abs(expected_distance - actual_distance) / max(0.035, expected_distance + 0.02)
        yaw_error = abs(expected_yaw - actual_yaw) / max(0.060, expected_yaw + 0.03)
        consistency = float(np.clip(1.0 - 0.62 * translation_error - 0.38 * yaw_error, 0.0, 1.0))
        scan_clearance = self._scan_clearance_score(after)
        tof_left = self._tof_clearance_score(after, TOF_SENSOR_LATERAL_OFFSET_M)
        tof_right = self._tof_clearance_score(after, -TOF_SENSOR_LATERAL_OFFSET_M)
        camera_visible = self._camera_visible_opponent_target_score(team)
        fusion = self.sensor_fusion[team]
        alpha = 0.72
        fusion["wheel_imu_consistency"] = alpha * fusion["wheel_imu_consistency"] + (1.0 - alpha) * consistency
        fusion["scan_clearance"] = alpha * fusion["scan_clearance"] + (1.0 - alpha) * scan_clearance
        fusion["tof_front_left_clearance"] = alpha * fusion["tof_front_left_clearance"] + (1.0 - alpha) * tof_left
        fusion["tof_front_right_clearance"] = alpha * fusion["tof_front_right_clearance"] + (1.0 - alpha) * tof_right
        fusion["camera_target_visible"] = alpha * fusion["camera_target_visible"] + (1.0 - alpha) * camera_visible
        fusion["bumper_or_hard_contact"] = 1.0 if hard_contact else 0.0
        fusion["pushable_contact"] = 1.0 if push_contact else 0.0
        previous_linear, previous_angular = self.last_motion_command.get(team, (0.0, 0.0))
        if self.dt > 1e-6:
            linear_accel = abs(float(linear_speed) - previous_linear) / self.dt
            angular_accel = abs(float(angular_speed) - previous_angular) / self.dt
        else:
            linear_accel = 0.0
            angular_accel = 0.0
        self.last_motion_command[team] = (float(linear_speed), float(angular_speed))
        linear_excess = max(0.0, linear_accel - ACCEL_DRIFT_LINEAR_THRESHOLD) / ACCEL_DRIFT_LINEAR_THRESHOLD
        angular_excess = max(0.0, angular_accel - ACCEL_DRIFT_ANGULAR_THRESHOLD) / ACCEL_DRIFT_ANGULAR_THRESHOLD
        accel_drift_loss = (
            ACCEL_DRIFT_LOSS_SCALE
            * self.domain_params.drift_loss_scale
            * (0.62 * linear_excess + 0.38 * angular_excess)
        )
        if accel_drift_loss > 0.0:
            fusion["wheel_imu_consistency"] = max(0.0, fusion["wheel_imu_consistency"] - 0.35 * accel_drift_loss)
        if self.last_contact:
            return
        fused_quality = (
            0.40 * fusion["wheel_imu_consistency"] +
            0.25 * fusion["scan_clearance"] +
            0.20 * min(fusion["tof_front_left_clearance"], fusion["tof_front_right_clearance"]) +
            0.15 * fusion["camera_target_visible"]
        )
        confidence_delta = FUSION_CONFIDENCE_RECOVERY_GAIN * max(0.0, fused_quality - 0.45)
        if fused_quality < 0.24:
            confidence_delta -= FUSION_CONFIDENCE_DRIFT_LOSS * (0.24 - fused_quality) / 0.24
        if blocked or hard_contact:
            confidence_delta -= FUSION_HARD_CONTACT_LOSS
        if jammed_push:
            confidence_delta -= FUSION_JAMMED_PUSH_LOSS
        confidence_delta -= accel_drift_loss
        self.localization_confidence[team] = float(
            np.clip(self.localization_confidence[team] + confidence_delta, 0.05, 1.0)
        )
    def _boost_sensor_fusion_recovery(self, team: str):
        fusion = self.sensor_fusion[team]
        for key in ("wheel_imu_consistency", "scan_clearance", "tof_front_left_clearance", "tof_front_right_clearance"):
            fusion[key] = min(1.0, 0.60 * fusion[key] + 0.40)
        fusion["bumper_or_hard_contact"] = 0.0
    def _can_relocalize(self, team: str) -> bool:
        return self.elapsed - float(self.last_relocalization_time[team]) >= RECOVERY_COOLDOWN_S
    def _scan_clearance_score(self, pose: np.ndarray) -> float:
        margin = max(0.0, self._arena_footprint_margin(pose))
        nearest = min(margin, 0.45)
        for center, half_size in self.nav_blockers:
            dx = max(0.0, abs(float(pose[0]) - center[0]) - half_size[0])
            dy = max(0.0, abs(float(pose[1]) - center[1]) - half_size[1])
            nearest = min(nearest, math.hypot(dx, dy))
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            dx = max(0.0, abs(float(pose[0]) - center[0]) - half_size[0])
            dy = max(0.0, abs(float(pose[1]) - center[1]) - half_size[1])
            nearest = min(nearest, math.hypot(dx, dy))
        for center in self.pushable_obstacles.values():
            dx = max(0.0, abs(float(pose[0]) - float(center[0])) - PUSHABLE_OBSTACLE_HALF)
            dy = max(0.0, abs(float(pose[1]) - float(center[1])) - PUSHABLE_OBSTACLE_HALF)
            nearest = min(nearest, math.hypot(dx, dy))
        return float(np.clip(nearest / 0.45, 0.0, 1.0))
    def _tof_clearance_score(self, pose: np.ndarray, lateral_offset: float) -> float:
        yaw = float(pose[2])
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        origin = pose[:2] + lateral * lateral_offset + forward * (ROBOT_LENGTH * 0.38)
        samples = 8
        for index in range(1, samples + 1):
            distance = TOF_SENSOR_RANGE_M * index / samples
            point = origin + forward * distance
            candidate = np.array([point[0], point[1], yaw], dtype=np.float32)
            if self._pose_blocked(candidate):
                return float(np.clip(distance / TOF_SENSOR_RANGE_M, 0.0, 1.0))
        return 1.0
    def _camera_visible_opponent_target_score(self, team: str) -> float:
        opponent = self._opponent(team)
        candidates = [
            target for target in self.targets
            if target.owner == opponent and not target.knocked
        ]
        if not candidates:
            return 0.0
        return 1.0 if any(self._target_visible_in_camera(team, target) for target in candidates) else 0.0
    def _team_point(self, team: str, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float32) * team_frame_sign(team)
    def _team_vector(self, team: str, vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) * team_frame_sign(team)
    def _team_yaw(self, team: str, yaw: float) -> float:
        return wrap_angle(yaw if team == "yellow" else yaw + math.pi)
    def _canonical_target_flags(self, team: str, normal_targets: list[Target]) -> np.ndarray:
        opponent = self._opponent(team)
        def key(target: Target) -> tuple[int, float, float]:
            xy = self._team_point(team, np.asarray(target.xy, dtype=np.float32))
            owner_group = 0 if target.owner == opponent else 1
            return owner_group, round(float(xy[0]), 4), round(float(xy[1]), 4)
        ordered = sorted(normal_targets, key=key)
        return np.array([1.0 if target.knocked else 0.0 for target in ordered], dtype=np.float32)
    def _base_distance(self, team: str) -> float:
        opponent_base = BLUE_BASE_XY if team == "yellow" else YELLOW_BASE_XY
        return float(np.linalg.norm(self.poses[team][:2] - opponent_base))
    @staticmethod
    def _opponent(team: str) -> str:
        return "blue" if team == "yellow" else "yellow"
