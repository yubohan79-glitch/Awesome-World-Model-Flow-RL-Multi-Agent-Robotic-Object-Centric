from __future__ import annotations

import math

import numpy as np

from ._compat import gym, spaces
from .datatypes import Target

from .constants import (
    ARENA_SIZE,
    BASE_HIT_RADIUS,
    BASE_SHOOT_RANGE,
    BASE_TARGET_CONTACT_RADIUS,
    BLUE_BASE_TARGET_XY,
    BLUE_BASE_TARGET_YAW,
    BLUE_BASE_XY,
    BLUE_ROUTE,
    BLUE_START,
    HALF_ARENA,
    LASER_DWELL_REQUIRED_S,
    LASER_FIRE_COOLDOWN_S,
    NORMAL_TARGET_CONTACT_RADIUS,
    NORTH_MIDDLE_TARGET_X,
    PUSHABLE_CLEARANCE_MARGIN,
    PUSHABLE_OBSTACLE_HALF,
    PUSHABLE_OBSTACLE_STARTS,
    PUSHABLE_STEP_M,
    ROBOT_RADIUS,
    ROUTE_CLEARANCE,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    SIDE_GATE_TARGET_Y,
    SOUTH_MIDDLE_TARGET_X,
    TARGET_WALL_INSET,
    WALL_THICKNESS,
    YELLOW_BASE_TARGET_XY,
    YELLOW_BASE_TARGET_YAW,
    YELLOW_BASE_XY,
    YELLOW_START
)
from .geometry import active_base_armor_blockers, base_attack_pose_quality, base_hit_success_cap, inward_45deg_target_yaws, laser_accuracy_from_geometry, laser_origin_from_pose, normalized_laser_dwell_factor, robot_pushable_collision, route_pose, segment_intersects_aabb, shooting_range_limits, wrap_angle

class RoboCupVisionRLGymEnv(gym.Env):
    """Fast 2D rule environment for validating tactics before IsaacLab replay.

    Action: [linear_velocity, angular_velocity, fire_gate], each in [-1, 1].
    Observation: normalized robot states, armor counts, target flags, nearest target vector, base vector.
    """

    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.10, max_time_s: float = 180.0):
        super().__init__()
        self.dt = dt
        self.max_time_s = max_time_s
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(31, -np.inf, dtype=np.float32),
            high=np.full(31, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.nav_blockers = self._make_blockers(inflated=True)
        self.laser_blockers = self._make_blockers(inflated=False)
        self.reset()

    def _make_blockers(self, *, inflated: bool) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        margin = ROUTE_CLEARANCE if inflated else 0.0
        wall_span = ARENA_SIZE + WALL_THICKNESS * 2.0
        raw = [
            ((-(HALF_ARENA + WALL_THICKNESS * 0.5), 0.0), (WALL_THICKNESS, wall_span)),
            (((HALF_ARENA + WALL_THICKNESS * 0.5), 0.0), (WALL_THICKNESS, wall_span)),
            ((0.0, -(HALF_ARENA + WALL_THICKNESS * 0.5)), (wall_span, WALL_THICKNESS)),
            ((0.0, (HALF_ARENA + WALL_THICKNESS * 0.5)), (wall_span, WALL_THICKNESS)),
            ((-1.00, 0.0), (1.00, WALL_THICKNESS)),
            ((1.00, 0.0), (1.00, WALL_THICKNESS)),
            ((0.00, 1.25), (WALL_THICKNESS, 0.50)),
            ((0.00, -1.25), (WALL_THICKNESS, 0.50)),
        ]
        return [(center, (size[0] * 0.5 + margin, size[1] * 0.5 + margin)) for center, size in raw]

    def _make_targets(self) -> list[Target]:
        n = HALF_ARENA - TARGET_WALL_INSET
        target_yaws = inward_45deg_target_yaws()
        return [
            Target("T01_NorthMiddle", (NORTH_MIDDLE_TARGET_X, n), target_yaws["T01_NorthMiddle"], "normal", "blue"),
            Target("T02_NorthEast", (n, n), target_yaws["T02_NorthEast"], "normal", "blue"),
            Target("T03_WestAboveGate", (-n, SIDE_GATE_TARGET_Y), target_yaws["T03_WestAboveGate"], "normal", "blue"),
            Target("T04_WestBelowGate", (-n, -SIDE_GATE_TARGET_Y), target_yaws["T04_WestBelowGate"], "normal", "yellow"),
            Target("T05_EastAboveGate", (n, SIDE_GATE_TARGET_Y), target_yaws["T05_EastAboveGate"], "normal", "blue"),
            Target("T06_EastBelowGate", (n, -SIDE_GATE_TARGET_Y), target_yaws["T06_EastBelowGate"], "normal", "yellow"),
            Target("T07_SouthWest", (-n, -n), target_yaws["T07_SouthWest"], "normal", "yellow"),
            Target("T08_SouthMiddle", (SOUTH_MIDDLE_TARGET_X, -n), target_yaws["T08_SouthMiddle"], "normal", "yellow"),
            Target("BlueBaseTarget", tuple(BLUE_BASE_TARGET_XY), BLUE_BASE_TARGET_YAW, "base_blue", "blue"),
            Target("YellowBaseTarget", tuple(YELLOW_BASE_TARGET_XY), YELLOW_BASE_TARGET_YAW, "base_yellow", "yellow"),
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.yellow = YELLOW_START.copy()
        self.blue = BLUE_START.copy()
        self.targets = self._make_targets()
        self.pushable_obstacles = {
            name: value.copy() for name, value in PUSHABLE_OBSTACLE_STARTS.items()
        }
        self.armor = {"yellow": 4, "blue": 4}
        self.elapsed = 0.0
        self.last_fire = {"yellow": -99.0, "blue": -99.0}
        self.laser_locks = {
            "yellow": {"target": "", "start": -99.0},
            "blue": {"target": "", "start": -99.0},
        }
        self.winner: str | None = None
        self.last_contact = False
        self.last_shot_attempt: dict[str, dict[str, object]] = {"yellow": {}, "blue": {}}
        self.localization_confidence = 1.0
        self.rng = np.random.default_rng(seed)
        self._previous_blue_base_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.elapsed += self.dt
        reward = -0.01
        info: dict[str, object] = {}

        previous_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        blocked = self._apply_yellow_action(action)
        if blocked:
            reward -= 0.4
            self.localization_confidence = max(0.05, self.localization_confidence - 0.16)

        self.blue = route_pose(self.elapsed, BLUE_ROUTE)
        if self._resolve_robot_contact():
            reward -= 1.0
            self.localization_confidence = max(0.05, self.localization_confidence - 0.38)
            info["robot_contact"] = True

        if self.localization_confidence < 0.62:
            spinning_in_place = abs(float(action[1])) > 0.62 and abs(float(action[0])) < 0.18
            if spinning_in_place:
                self.localization_confidence = min(1.0, self.localization_confidence + 0.14)
                reward += 0.35
                info["relocalizing"] = True
            else:
                reward -= 0.08

        if action[2] > 0.25 and self.elapsed - self.last_fire["yellow"] > LASER_FIRE_COOLDOWN_S:
            reward += self._apply_fire_rule("yellow", info)
        else:
            self._reset_laser_lock("yellow")

        if self.elapsed - self.last_fire["blue"] > 1.4:
            self._apply_fire_rule("blue", info)

        new_distance = float(np.linalg.norm(self.yellow[:2] - BLUE_BASE_XY))
        reward += 0.15 * (previous_distance - new_distance)
        terminated = self.winner is not None
        truncated = self.elapsed >= self.max_time_s
        if self.winner == "yellow":
            reward += 60.0
        elif self.winner == "blue":
            reward -= 45.0
        return self._get_obs(), float(reward), terminated, truncated, info

    def _apply_yellow_action(self, action: np.ndarray) -> bool:
        candidate = self.yellow.copy()
        linear_speed = 0.45 * float(action[0])
        angular_speed = 1.8 * float(action[1])
        candidate[2] = wrap_angle(float(candidate[2] + angular_speed * self.dt))
        candidate[0] += linear_speed * math.cos(float(candidate[2])) * self.dt
        candidate[1] += linear_speed * math.sin(float(candidate[2])) * self.dt
        if self._static_pose_blocked(candidate):
            return True
        if self._target_collision_name(candidate) is not None:
            return True
        obstacle_name = self._pushable_collision_name(candidate)
        if obstacle_name is not None:
            motion_yaw = float(candidate[2]) if linear_speed > 0.0 else wrap_angle(float(candidate[2]) + math.pi)
            if abs(linear_speed) <= 0.03 or not self._push_obstacle(obstacle_name, motion_yaw, candidate[:2]):
                return True
            if self._pushable_collision_name(candidate) is not None:
                return True
        self.yellow = candidate
        return False

    def _pose_blocked(self, pose: np.ndarray) -> bool:
        return (
            self._static_pose_blocked(pose)
            or self._pushable_collision_name(pose) is not None
            or self._target_collision_name(pose) is not None
        )

    def _static_pose_blocked(self, pose: np.ndarray) -> bool:
        x, y = float(pose[0]), float(pose[1])
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
                return True
        return False

    def _pushable_collision_name(self, pose: np.ndarray) -> str | None:
        for name, center in self.pushable_obstacles.items():
            collided, _normal, _penetration = robot_pushable_collision(
                pose,
                (float(center[0]), float(center[1])),
            )
            if collided:
                return name
        return None

    def _pushable_position_valid(self, obstacle_name: str, xy: np.ndarray, robot_pose: np.ndarray) -> bool:
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSHABLE_CLEARANCE_MARGIN
        if abs(float(xy[0])) > limit or abs(float(xy[1])) > limit:
            return False
        inflated = PUSHABLE_OBSTACLE_HALF + PUSHABLE_CLEARANCE_MARGIN
        for center, half_size in self.nav_blockers:
            if abs(float(xy[0]) - center[0]) <= half_size[0] + inflated and abs(float(xy[1]) - center[1]) <= half_size[1] + inflated:
                return False
        for name, center in self.pushable_obstacles.items():
            if name == obstacle_name:
                continue
            if float(np.linalg.norm(xy - center)) < PUSHABLE_OBSTACLE_HALF * 2.0 + PUSHABLE_CLEARANCE_MARGIN:
                return False
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            if float(np.linalg.norm(xy - np.asarray(target.xy, dtype=np.float32))) < inflated + radius:
                return False
        pose = robot_pose if robot_pose.shape[0] >= 3 else np.array([robot_pose[0], robot_pose[1], 0.0], dtype=np.float32)
        collided, _normal, _penetration = robot_pushable_collision(
            pose,
            (float(xy[0]), float(xy[1])),
        )
        return not collided

    def _target_collision_name(self, pose: np.ndarray) -> str | None:
        x, y = float(pose[0]), float(pose[1])
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            if (target.xy[0] - x) ** 2 + (target.xy[1] - y) ** 2 <= (ROBOT_RADIUS + radius) ** 2:
                return target.name
        return None

    def _push_obstacle(self, obstacle_name: str, motion_yaw: float, robot_xy: np.ndarray) -> bool:
        current = self.pushable_obstacles[obstacle_name]
        direction = np.array([math.cos(motion_yaw), math.sin(motion_yaw)], dtype=np.float32)
        limit = HALF_ARENA - PUSHABLE_OBSTACLE_HALF - PUSHABLE_CLEARANCE_MARGIN
        accepted = None
        robot_pose = np.array([float(robot_xy[0]), float(robot_xy[1]), motion_yaw], dtype=np.float32)
        for multiplier in (1.0, 1.7, 2.4, 3.1, 4.0):
            candidate = current + direction * (PUSHABLE_STEP_M * multiplier)
            candidate = np.array(
                [
                    float(np.clip(candidate[0], -limit, limit)),
                    float(np.clip(candidate[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            if self._pushable_position_valid(obstacle_name, candidate, robot_pose):
                accepted = candidate
                break
        if accepted is None:
            return False
        self.pushable_obstacles[obstacle_name] = accepted
        return True

    def _resolve_robot_contact(self) -> bool:
        delta = self.blue[:2] - self.yellow[:2]
        distance = float(np.linalg.norm(delta))
        min_distance = ROBOT_RADIUS * 2.0
        if distance >= min_distance:
            self.last_contact = False
            return False
        normal = np.array([1.0, 0.0], dtype=np.float32) if distance < 1e-6 else delta / distance
        push = (min_distance - max(distance, 1e-6)) * 0.5 + 0.004
        self.yellow[:2] -= normal * push
        self.blue[:2] += normal * push
        self.last_contact = True
        return True

    def _detect_laser_hit(self, team: str) -> Target | None:
        pose = self.yellow if team == "yellow" else self.blue
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
            if self._line_blocked(origin, target.xy):
                continue
            if target.owner == team:
                own_candidate_projection = min(own_candidate_projection, projection)
                continue
            accuracy = laser_accuracy_from_geometry(projection, perpendicular, target.kind.startswith("base_"))
            if target.kind.startswith("base_"):
                opponent = "blue" if team == "yellow" else "yellow"
                normal_hits = max(0, 4 - int(self.armor[opponent]))
                base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
                pose_quality = base_attack_pose_quality(normal_hits, target.xy, target.yaw, base_xy, pose[:2])
                if pose_quality <= 0.0:
                    continue
                accuracy = min(base_hit_success_cap(normal_hits), accuracy * pose_quality)
            if projection < best_projection:
                best_projection = projection
                best_target = target
                best_accuracy = accuracy
                best_lateral_error = perpendicular
        if own_candidate_projection <= best_projection:
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
        final_accuracy = float(np.clip(best_accuracy * dwell_factor, 0.0, 0.95))
        hit = bool(self.rng.random() <= final_accuracy)
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
        }
        self._reset_laser_lock(team)
        return best_target if hit else None

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

    def _opponent_tracking_features(self) -> np.ndarray:
        delta = self.blue[:2] - self.yellow[:2]
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-6 else float(self.yellow[2])
        relative_bearing = wrap_angle(bearing - float(self.yellow[2]))
        visible = 0.0 if self._line_blocked((float(self.yellow[0]), float(self.yellow[1])), (float(self.blue[0]), float(self.blue[1]))) else 1.0

        base_delta = YELLOW_BASE_XY - self.blue[:2]
        base_distance = float(np.linalg.norm(base_delta))
        base_bearing = math.atan2(float(base_delta[1]), float(base_delta[0])) if base_distance > 1e-6 else float(self.blue[2])
        heading_to_yellow_base = abs(wrap_angle(base_bearing - float(self.blue[2])))
        proximity_threat = max(0.0, 1.0 - base_distance / 1.10)
        heading_threat = max(0.0, 1.0 - heading_to_yellow_base / math.pi)
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

    def _apply_fire_rule(self, team: str, info: dict[str, object]) -> float:
        target = self._detect_laser_hit(team)
        info[f"{team}_shot_attempt"] = self.last_shot_attempt.get(team, {})
        if target is None:
            reason = str(self.last_shot_attempt.get(team, {}).get("reason", ""))
            if reason == "dwell":
                return 0.015 if team == "yellow" else 0.0
            if reason == "probabilistic_miss":
                self.last_fire[team] = self.elapsed
            return -0.05 if team == "yellow" else 0.0
        self.last_fire[team] = self.elapsed
        opponent = "blue" if team == "yellow" else "yellow"
        if target.kind == f"base_{team}":
            info[f"{team}_own_base_blocked"] = target.name
            return -1.0 if team == "yellow" else 0.0
        if target.owner == team:
            info[f"{team}_own_target_blocked"] = target.name
            return -1.0 if team == "yellow" else 0.0
        if target.kind == "normal":
            target.knocked = True
            self.armor[opponent] = max(0, self.armor[opponent] - 1)
            info[f"{team}_hit"] = target.name
            return 6.0 if team == "yellow" else 0.0
        if target.kind == f"base_{opponent}":
            target.knocked = True
            self.winner = team
            info["winner"] = team
            return 70.0 if team == "yellow" else 0.0
        return -0.1 if team == "yellow" else 0.0

    def _get_obs(self) -> np.ndarray:
        normal_targets = [target for target in self.targets if target.kind == "normal"]
        active_normals = [target for target in normal_targets if not target.knocked and target.owner != "yellow"]
        if active_normals:
            nearest = min(active_normals, key=lambda t: np.linalg.norm(np.array(t.xy, dtype=np.float32) - self.yellow[:2]))
            nearest_vec = (np.array(nearest.xy, dtype=np.float32) - self.yellow[:2]) / ARENA_SIZE
        else:
            nearest_vec = np.zeros(2, dtype=np.float32)
        blue_base_vec = (BLUE_BASE_XY - self.yellow[:2]) / ARENA_SIZE
        knocked_flags = np.array([1.0 if target.knocked else 0.0 for target in normal_targets], dtype=np.float32)
        opponent_track = self._opponent_tracking_features()
        obs = np.concatenate(
            [
                np.array([self.yellow[0] / HALF_ARENA, self.yellow[1] / HALF_ARENA, math.cos(self.yellow[2]), math.sin(self.yellow[2])]),
                np.array([self.blue[0] / HALF_ARENA, self.blue[1] / HALF_ARENA, math.cos(self.blue[2]), math.sin(self.blue[2])]),
                opponent_track,
                np.array([self.armor["blue"] / 4.0, self.armor["yellow"] / 4.0, self.elapsed / self.max_time_s, float(self.last_contact)]),
                np.array([self.localization_confidence]),
                knocked_flags,
                nearest_vec,
                blue_base_vec,
                np.array([1.0 if self.winner == "yellow" else -1.0 if self.winner == "blue" else 0.0]),
            ]
        ).astype(np.float32)
        return obs


if __name__ == "__main__":
    env = RoboCupVisionRLGymEnv()
    obs, _ = env.reset(seed=7)
    for _ in range(16):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"reward={reward:.3f} done={terminated or truncated} info={info}")
        if terminated or truncated:
            break
