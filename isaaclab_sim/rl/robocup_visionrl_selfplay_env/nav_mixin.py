from __future__ import annotations

import math
import heapq

import numpy as np

from .constants import (
    BASE_ATTACK_STALE_STEP_LIMIT,
    BASE_FIRE_HOLD_RADIUS,
    BASE_FIRE_REPLAN_NUDGE_M,
    BASE_RUSH_PREFERRED_NORMAL_HITS,
    BASE_TARGET_CONTACT_RADIUS,
    FIRE_POSE_BLOCKED_STEP_LIMIT,
    NORMAL_ATTACK_STALE_STEP_LIMIT,
    NORMAL_FIRE_HOLD_RADIUS,
    NORMAL_TARGET_CONTACT_RADIUS,
    POST_HIT_RETREAT_SPEED,
    PUSH_INTENT_THRESHOLD,
    RECOVERY_CONFIDENCE_THRESHOLD,
    TACTICAL_ACTION_LABELS
)
from robocup_visionrl_gym_env import (
    BLUE_BASE_XY,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_LENGTH,
    ROBOT_PUSHABLE_CLEARANCE_RADIUS,
    ROBOT_RADIUS,
    ROBOT_WIDTH,
    YELLOW_BASE_XY,
    active_base_armor_blockers,
    segment_intersects_aabb,
    shooting_range_limits,
    wrap_angle
)


class NavMixin:
    def _apply_action(self, team: str, action: np.ndarray | None) -> tuple[bool, dict[str, object]]:
        action = self._coerce_action(action)
        action, contact_shielded = self._shield_contact_action(team, action)
        self.pending_fire[team] = False
        self.selected_target_name[team] = None
        info: dict[str, object] = {
            "action_labels": TACTICAL_ACTION_LABELS,
        }
        if contact_shielded:
            info["action_shield_contact"] = True
        if (
            self.localization_confidence[team] < RECOVERY_CONFIDENCE_THRESHOLD
            and float(action[3]) > 0.35
            and self._can_relocalize(team)
        ):
            spin = 0.72 if team == "yellow" else -0.72
            blocked = self._integrate_command(team, 0.0, spin)
            self._reset_laser_lock(team)
            self.strategy_counts[team]["recovery_steps"] += 1
            info.update({"tactic": "recover", "recovery_gate": float(action[3])})
            return blocked, info
        if self.elapsed < self.post_hit_retreat_until[team]:
            blocked = self._integrate_command(team, POST_HIT_RETREAT_SPEED, 0.0, allow_push=True)
            self._reset_laser_lock(team)
            info.update(
                {
                    "tactic": "push_clear",
                    "post_hit_retreat": True,
                    "pushed_obstacle": self.last_push_event[team],
                    "push_impulse": self.last_push_impulse[team],
                }
            )
            return blocked, info
        if self._should_block(team, action):
            risk = (float(action[5]) + 1.0) * 0.5
            goal = self._block_goal(team, risk)
            opponent = self._opponent(team)
            blocked = self._drive_to_goal(team, goal, risk, face_xy=self.poses[opponent][:2])
            self.strategy_counts[team]["block_steps"] += 1
            if risk > 0.72:
                self.strategy_counts[team]["interference_steps"] += 1
            self._reset_laser_lock(team)
            info.update(
                {
                    "tactic": "block",
                    "goal_xy": tuple(round(float(v), 3) for v in goal),
                    "opponent_threat": self._opponent_threat(team),
                    "interference": risk > 0.72,
                    "pushed_obstacle": self.last_push_event[team],
                    "push_impulse": self.last_push_impulse[team],
                }
            )
            return blocked, info
        target = self._select_tactical_target(team, action)
        if target is None:
            info["tactic"] = "wait"
            self._reset_laser_lock(team)
            return self._integrate_command(team, 0.0, 0.0), info
        risk = (float(action[5]) + 1.0) * 0.5
        goal = self._fire_standoff_goal(team, target, risk)
        pre_goal_distance = float(np.linalg.norm(goal - self.poses[team][:2]))
        geometry_snapshot = self._fire_geometry_snapshot(team, target, risk)
        center_distance = float(geometry_snapshot["center_distance"])
        line_clear = bool(geometry_snapshot["line_clear"])
        yaw_error = float(geometry_snapshot["yaw_error"])
        transient_base_occlusion = (
            target.kind.startswith("base_")
            and yaw_error > 0.25
            and self._center_aim_line_clear(team, target)
        )
        base_target = target.kind.startswith("base_")
        hold_radius = BASE_FIRE_HOLD_RADIUS if base_target else NORMAL_FIRE_HOLD_RADIUS
        min_shot_range, max_shot_range = shooting_range_limits(base_target)
        shot_distance_now = float(geometry_snapshot["shot_distance"])
        line_ok_for_hold = (
            line_clear
            or transient_base_occlusion
            or (base_target and pre_goal_distance < hold_radius and bool(geometry_snapshot["base_pose_ok"]))
        )
        near_fire_window = (
            pre_goal_distance < hold_radius
            and line_ok_for_hold
            and center_distance >= self._target_contact_clearance(target)
            and shot_distance_now >= min_shot_range + 0.006
            and shot_distance_now <= max_shot_range
        )
        holding_fire_pose = bool(geometry_snapshot["geometry_ready"]) or near_fire_window
        geometry_ready = bool(geometry_snapshot["geometry_ready"])
        if holding_fire_pose and base_target and not geometry_ready and pre_goal_distance < hold_radius:
            refined = self._best_fire_pose(team, target, risk, route_aware=True)
            if refined is not None and float(np.linalg.norm(refined[0] - self.poses[team][:2])) > BASE_FIRE_REPLAN_NUDGE_M:
                goal = refined[0]
                blocked = self._drive_to_goal(team, goal, risk, face_xy=np.asarray(target.xy, dtype=np.float32))
                holding_fire_pose = False
            else:
                blocked = self._hold_fire_pose(team, target, risk)
        elif holding_fire_pose:
            blocked = self._hold_fire_pose(team, target, risk)
            if blocked and not geometry_ready:
                face_target_xy = None if not line_clear else np.asarray(target.xy, dtype=np.float32)
                blocked = self._drive_to_goal(team, goal, risk, face_xy=face_target_xy)
                holding_fire_pose = False
        else:
            face_target_xy = np.asarray(target.xy, dtype=np.float32)
            if target.kind.startswith("base_") and pre_goal_distance > 0.075:
                face_target_xy = None
            if not bool(geometry_snapshot["line_clear"]) and pre_goal_distance > 0.030:
                face_target_xy = None
            blocked = self._drive_to_goal(team, goal, risk, face_xy=face_target_xy)
        fire_info = self._update_fire_gate(team, target, action, risk)
        fire_pose_replanned = False
        if holding_fire_pose and blocked and not self.pending_fire[team]:
            blocked_steps = self.fire_pose_blocked_steps[team].get(target.name, 0) + 1
            self.fire_pose_blocked_steps[team][target.name] = blocked_steps
            if blocked_steps >= FIRE_POSE_BLOCKED_STEP_LIMIT:
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.fire_pose_blocked_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                fire_pose_replanned = True
        else:
            self.fire_pose_blocked_steps[team].pop(target.name, None)
        normal_attack_replanned = False
        if target.kind == "normal" and not self.pending_fire[team]:
            normal_stale_steps = self.normal_attack_stale_steps[team].get(target.name, 0) + 1
            self.normal_attack_stale_steps[team][target.name] = normal_stale_steps
            if normal_stale_steps >= NORMAL_ATTACK_STALE_STEP_LIMIT:
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.normal_attack_stale_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                normal_attack_replanned = True
        else:
            self.normal_attack_stale_steps[team].pop(target.name, None)
        base_attack_replanned = False
        if target.kind.startswith("base_") and not self.pending_fire[team] and self._normal_hits_against(team) < 4:
            stale_steps = self.base_attack_stale_steps[team].get(target.name, 0) + 1
            self.base_attack_stale_steps[team][target.name] = stale_steps
            if stale_steps >= BASE_ATTACK_STALE_STEP_LIMIT:
                normal_hits = self._normal_hits_against(team)
                required_hits = min(4, normal_hits + 1) if normal_hits < BASE_RUSH_PREFERRED_NORMAL_HITS else normal_hits
                self.base_retry_min_normal_hits[team] = max(
                    int(self.base_retry_min_normal_hits.get(team, 0)),
                    required_hits,
                )
                self._mark_target_failed(team, target.name)
                self._reset_laser_lock(team)
                self.base_attack_stale_steps[team].pop(target.name, None)
                self._fire_pose_cache.clear()
                base_attack_replanned = True
        else:
            self.base_attack_stale_steps[team].pop(target.name, None)
        self.selected_target_name[team] = target.name
        self.strategy_counts[team]["attack_steps"] += 1
        if target.kind.startswith("base_"):
            self.strategy_counts[team]["base_rush_steps"] += 1
        info.update(
            {
                "tactic": "attack",
                "selected_target": target.name,
                "base_rush": target.kind.startswith("base_"),
                "goal_xy": tuple(round(float(v), 3) for v in goal),
                "goal_distance_m": round(float(np.linalg.norm(goal - self.poses[team][:2])), 4),
                "fire_ready": self.pending_fire[team],
                "holding_fire_pose": holding_fire_pose,
                "fire_pose_replanned": fire_pose_replanned,
                "normal_attack_replanned": normal_attack_replanned,
                "base_attack_replanned": base_attack_replanned,
                "pushed_obstacle": self.last_push_event[team],
                "push_impulse": self.last_push_impulse[team],
            }
        )
        info.update(fire_info)
        return blocked, info
    def _integrate_command(
        self,
        team: str,
        linear_speed: float,
        angular_speed: float,
        *,
        allow_push: bool = False,
    ) -> bool:
        self.last_push_event[team] = ""
        self.last_push_impulse[team] = {}
        before = self.poses[team].copy()
        separated_before = self._separated_pose_from_all_pushables(before)
        if not np.allclose(separated_before, before, atol=1e-6):
            self.poses[team] = separated_before
            before = separated_before.copy()
        linear_speed = float(linear_speed) * self.domain_params.drive_scale
        angular_speed = float(angular_speed) * self.domain_params.turn_scale
        pose = before.copy()
        pose[2] = wrap_angle(float(pose[2] + angular_speed * self.dt))
        pose[0] += linear_speed * math.cos(float(pose[2])) * self.dt
        pose[1] += linear_speed * math.sin(float(pose[2])) * self.dt
        if self._static_pose_blocked(pose):
            if abs(linear_speed) <= 1e-6 and not self._footprint_outside_arena(pose):
                self.poses[team] = pose
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    pose,
                    linear_speed,
                    angular_speed,
                    blocked=False,
                )
                return False
            self._record_motion_sensor_fusion(
                team,
                before,
                before,
                linear_speed,
                angular_speed,
                blocked=True,
                hard_contact=True,
            )
            return True
        if self._target_collision_name(pose) is not None:
            self._record_motion_sensor_fusion(
                team,
                before,
                before,
                linear_speed,
                angular_speed,
                blocked=True,
                hard_contact=True,
            )
            return True
        obstacle_name = self._pushable_collision_name(pose)
        if obstacle_name is not None:
            if abs(linear_speed) <= 0.03:
                separated = self._separated_pose_from_pushable(pose, obstacle_name)
                if self._pushable_collision_name(separated) is None:
                    self.poses[team] = separated
                    self._record_motion_sensor_fusion(
                        team,
                        before,
                        separated,
                        linear_speed,
                        angular_speed,
                        blocked=False,
                        push_contact=True,
                    )
                    return False
            if not allow_push or abs(linear_speed) <= 0.03:
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
            motion_yaw = float(pose[2]) if linear_speed > 0.0 else wrap_angle(float(pose[2]) + math.pi)
            if not self._push_obstacle(team, obstacle_name, motion_yaw, pose[:2]):
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
            pose = self._apply_push_recoil_pose(pose, motion_yaw, linear_speed)
            pose = self._separated_pose_from_pushable(pose, obstacle_name)
            if self._pushable_collision_name(pose) is not None:
                self._record_motion_sensor_fusion(
                    team,
                    before,
                    before,
                    linear_speed,
                    angular_speed,
                    blocked=True,
                    push_contact=True,
                    jammed_push=True,
                )
                return True
        self.poses[team] = pose
        self._record_motion_sensor_fusion(
            team,
            before,
            pose,
            linear_speed,
            angular_speed,
            blocked=False,
            push_contact=obstacle_name is not None,
        )
        return False
    def _drive_to_goal(self, team: str, goal_xy: np.ndarray, risk: float, face_xy: np.ndarray | None = None) -> bool:
        pose = self.poses[team]
        subgoal_xy, final_leg = self._planned_subgoal(pose[:2], goal_xy)
        for _ in range(2):
            if self._local_blocker_cost(pose[:2]) > 0.25:
                break
            if final_leg or float(np.linalg.norm(subgoal_xy - pose[:2])) >= 0.10:
                break
            next_subgoal, next_final = self._planned_subgoal(subgoal_xy, goal_xy)
            if np.allclose(next_subgoal, subgoal_xy, atol=1e-4):
                break
            subgoal_xy, final_leg = next_subgoal, next_final
        dx = float(subgoal_xy[0] - pose[0])
        dy = float(subgoal_xy[1] - pose[1])
        distance = math.hypot(dx, dy)
        if final_leg and face_xy is not None and distance < 0.003:
            desired_yaw = math.atan2(float(face_xy[1] - pose[1]), float(face_xy[0] - pose[0]))
        else:
            desired_yaw = math.atan2(dy, dx) if distance > 1e-6 else float(pose[2])
        yaw_error = wrap_angle(desired_yaw - float(pose[2]))
        angular_speed = float(np.clip(3.05 * yaw_error, -2.65, 2.65))
        alignment = max(0.0, 1.0 - abs(yaw_error) / 1.20)
        max_speed = 0.22 + 0.22 * risk
        linear_speed = max_speed * max(0.10, alignment)
        stop_radius = 0.002 if final_leg else 0.035
        if distance < stop_radius:
            linear_speed = 0.0
        elif final_leg:
            linear_speed = min(linear_speed, max(0.0, (distance - stop_radius) * 0.70 / max(self.dt, 1e-6)))
        near_blocker = self._local_blocker_cost(pose[:2]) > 0.25
        cautious_turning = near_blocker or distance < 0.24
        hard_turn_limit = 0.78 if cautious_turning else 1.35
        slow_turn_limit = 0.45 if cautious_turning else 0.95
        if abs(yaw_error) > hard_turn_limit:
            linear_speed = 0.0
            escape_speed = self._boundary_escape_linear_speed(pose)
            if abs(escape_speed) > 1e-6:
                linear_speed = escape_speed
        elif abs(yaw_error) > slow_turn_limit:
            linear_speed *= 0.25
        blocked = self._integrate_command(team, linear_speed, angular_speed, allow_push=risk >= PUSH_INTENT_THRESHOLD)
        if blocked and linear_speed <= 0.02 and distance > stop_radius:
            # Differential-drive escape: when a corner/armor footprint rejects
            # in-place rotation, back out slowly instead of staying locked.
            escape_speed = -0.11 * max(0.45, 1.0 - self._arena_footprint_margin(pose) / 0.16)
            escape_turn = 0.55 * angular_speed
            escaped = self._integrate_command(team, escape_speed, escape_turn, allow_push=False)
            if not escaped:
                return False
        if blocked and linear_speed > 0.0:
            for scale in (0.45, 0.30, 0.20, 0.13):
                cautious_speed = max(0.045, linear_speed * scale)
                cautious = self._integrate_command(
                    team,
                    cautious_speed,
                    angular_speed,
                    allow_push=risk >= PUSH_INTENT_THRESHOLD,
                )
                if not cautious:
                    return False
            return self._integrate_command(team, 0.0, angular_speed, allow_push=False)
        return blocked
    def _planned_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        corridor_xy, corridor_final = self._central_lane_subgoal(current_xy, goal_xy)
        if not corridor_final:
            if not self._segment_blocked_for_nav(current_xy, corridor_xy):
                return corridor_xy, False
            path_to_corridor = self._astar_path(current_xy, corridor_xy)
            if len(path_to_corridor) >= 3:
                return path_to_corridor[2], False
            if len(path_to_corridor) == 2:
                return path_to_corridor[1], False
        if not self._segment_blocked_for_nav(current_xy, goal_xy):
            return goal_xy, True
        path = self._astar_path(current_xy, goal_xy)
        if len(path) >= 3:
            return path[2], False
        if len(path) == 2:
            return path[1], False
        return self._corridor_subgoal(current_xy, goal_xy)
    def _central_lane_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        current_y = float(current_xy[1])
        goal_y = float(goal_xy[1])
        if current_y < -0.06 and goal_y > 0.06:
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y < -0.35:
                return np.array([gate_x, -0.24], dtype=np.float32), False
            return np.array([gate_x, 0.24], dtype=np.float32), False
        if (
            -0.06 <= current_y < 0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y > 0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            return np.array([gate_x, 0.36], dtype=np.float32), False
        if current_y > 0.06 and goal_y < -0.06:
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y > 0.35:
                return np.array([gate_x, 0.24], dtype=np.float32), False
            return np.array([gate_x, -0.24], dtype=np.float32), False
        if (
            0.06 >= current_y > -0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y < -0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            return np.array([gate_x, -0.36], dtype=np.float32), False
        return goal_xy, True
    def _segment_blocked_for_nav(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        distance = float(np.linalg.norm(goal_xy - start_xy))
        samples = max(2, int(math.ceil(distance / 0.06)))
        for index in range(1, samples + 1):
            alpha = index / samples
            point = start_xy * (1.0 - alpha) + goal_xy * alpha
            if self._point_blocked_for_nav(float(point[0]), float(point[1])):
                return True
        origin = (float(start_xy[0]), float(start_xy[1]))
        target = (float(goal_xy[0]), float(goal_xy[1]))
        push_half = PUSHABLE_OBSTACLE_HALF + ROBOT_PUSHABLE_CLEARANCE_RADIUS
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(origin, target, (float(center[0]), float(center[1])), (push_half, push_half)):
                return True
        return False
    def _segment_likely_blocked_for_nav(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        origin = (float(start_xy[0]), float(start_xy[1]))
        target = (float(goal_xy[0]), float(goal_xy[1]))
        for center, half_size in self.nav_blockers:
            if segment_intersects_aabb(origin, target, center, half_size):
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if segment_intersects_aabb(origin, target, center, half_size):
                return True
        push_half = PUSHABLE_OBSTACLE_HALF + ROBOT_PUSHABLE_CLEARANCE_RADIUS
        for center in self.pushable_obstacles.values():
            if segment_intersects_aabb(origin, target, (float(center[0]), float(center[1])), (push_half, push_half)):
                return True
        return False
    def _astar_path(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray]:
        resolution = 0.18
        limit = HALF_ARENA - ROBOT_RADIUS - 0.02
        def to_key(point: np.ndarray) -> tuple[int, int]:
            clamped = np.array(
                [
                    float(np.clip(point[0], -limit, limit)),
                    float(np.clip(point[1], -limit, limit)),
                ],
                dtype=np.float32,
            )
            return (int(round(float(clamped[0]) / resolution)), int(round(float(clamped[1]) / resolution)))
        def to_point(key: tuple[int, int]) -> np.ndarray:
            return np.array(
                [
                    float(np.clip(key[0] * resolution, -limit, limit)),
                    float(np.clip(key[1] * resolution, -limit, limit)),
                ],
                dtype=np.float32,
            )
        def free(key: tuple[int, int]) -> bool:
            point = to_point(key)
            return not self._point_blocked_for_nav(float(point[0]), float(point[1]))
        start_key = to_key(self._nearest_free_xy(current_xy))
        goal_key = to_key(self._nearest_free_xy(goal_xy))
        obstacle_signature = tuple(
            sorted((round(float(v[0]) / resolution), round(float(v[1]) / resolution)) for v in self.pushable_obstacles.values())
        )
        cache_key = (start_key, goal_key, obstacle_signature)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        min_cell = int(math.floor(-limit / resolution))
        max_cell = int(math.ceil(limit / resolution))
        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_key))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far: dict[tuple[int, int], float] = {start_key: 0.0}
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.42),
            (-1, 1, 1.42),
            (1, -1, 1.42),
            (1, 1, 1.42),
        ]
        while open_heap:
            _priority, current = heapq.heappop(open_heap)
            if current == goal_key:
                break
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt[0] < min_cell or nxt[0] > max_cell or nxt[1] < min_cell or nxt[1] > max_cell:
                    continue
                if not free(nxt):
                    continue
                if dx != 0 and dy != 0 and (not free((current[0] + dx, current[1])) or not free((current[0], current[1] + dy))):
                    continue
                point = to_point(nxt)
                new_cost = cost_so_far[current] + step_cost + 0.04 * self._local_blocker_cost(point)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heuristic = math.hypot(goal_key[0] - nxt[0], goal_key[1] - nxt[1])
                    heapq.heappush(open_heap, (new_cost + heuristic, nxt))
                    came_from[nxt] = current
        if goal_key not in came_from and goal_key != start_key:
            self._path_cache[cache_key] = []
            return []
        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(came_from[keys[-1]])
        keys.reverse()
        path = [to_point(key) for key in keys]
        if len(path) > 0:
            path[-1] = self._nearest_free_xy(goal_xy)
        self._path_cache[cache_key] = path
        if len(self._path_cache) > 512:
            self._path_cache.clear()
        return path
    def _corridor_subgoal(self, current_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        current_y = float(current_xy[1])
        goal_y = float(goal_xy[1])
        if current_y < -0.06 and goal_y > 0.06:
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y < -0.35:
                return np.array([gate_x, -0.24], dtype=np.float32), False
            return np.array([gate_x, 0.24], dtype=np.float32), False
        if (
            -0.06 <= current_y < 0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y > 0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = 0.28 if float(current_xy[0]) >= 0.0 else -0.28
            return np.array([gate_x, 0.36], dtype=np.float32), False
        if current_y > 0.06 and goal_y < -0.06:
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            if abs(float(current_xy[0]) - gate_x) > 0.10 or current_y > 0.35:
                return np.array([gate_x, 0.24], dtype=np.float32), False
            return np.array([gate_x, -0.24], dtype=np.float32), False
        if (
            0.06 >= current_y > -0.30
            and abs(float(current_xy[0])) < 0.45
            and goal_y < -0.06
            and abs(float(goal_xy[0])) > 0.55
        ):
            gate_x = -0.28 if float(current_xy[0]) <= 0.0 else 0.28
            return np.array([gate_x, -0.36], dtype=np.float32), False
        if abs(current_y) < 0.18 and abs(float(goal_xy[0])) > 0.34 and abs(float(current_xy[0])) < 0.38:
            return np.array([float(current_xy[0]), 0.24 if goal_y >= 0.0 else -0.24], dtype=np.float32), False
        if goal_y > 0.80 and float(goal_xy[0]) < -0.55 and (
            float(current_xy[0]) > -0.58 or current_y > 0.36
        ):
            if current_y > 0.36:
                return np.array([float(current_xy[0]), 0.30], dtype=np.float32), False
            return np.array([-0.62, 0.30], dtype=np.float32), False
        if goal_y < -0.80 and float(goal_xy[0]) > 0.55 and (
            float(current_xy[0]) < 0.58 or current_y < -0.36
        ):
            if current_y < -0.36:
                return np.array([float(current_xy[0]), -0.30], dtype=np.float32), False
            return np.array([0.62, -0.30], dtype=np.float32), False
        if float(goal_xy[0]) > 0.90 and goal_y > 0.50 and current_y < 1.04:
            if float(current_xy[0]) < 1.14 or current_y < 0.26:
                return np.array([1.18, 0.30], dtype=np.float32), False
            return np.array([1.18, 1.08], dtype=np.float32), False
        if float(goal_xy[0]) < -0.90 and goal_y > 0.50 and current_y < 1.04:
            if float(current_xy[0]) > -1.14 or current_y < 0.26:
                return np.array([-1.18, 0.30], dtype=np.float32), False
            return np.array([-1.18, 1.08], dtype=np.float32), False
        if float(goal_xy[0]) > 0.90 and goal_y < -0.50 and current_y > -1.04:
            if float(current_xy[0]) < 1.14 or current_y > -0.26:
                return np.array([1.18, -0.30], dtype=np.float32), False
            return np.array([1.18, -1.08], dtype=np.float32), False
        if float(goal_xy[0]) < -0.90 and goal_y < -0.50 and current_y > -1.04:
            if float(current_xy[0]) > -1.14 or current_y > -0.26:
                return np.array([-1.18, -0.30], dtype=np.float32), False
            return np.array([-1.18, -1.08], dtype=np.float32), False
        return goal_xy, True
    def _should_block(self, team: str, action: np.ndarray) -> bool:
        opponent = self._opponent(team)
        time_remaining = max(0.0, self.max_time_s - self.elapsed)
        score_delta = self.scores[team] - self.scores[opponent]
        threat = self._opponent_threat(team)
        block_gate = float(action[2])
        risk = (float(action[5]) + 1.0) * 0.5
        if block_gate <= 0.25:
            return False
        if threat > 0.68 and block_gate > 0.45:
            return True
        if score_delta >= 10 and time_remaining < 35.0 and block_gate > 0.55:
            return True
        opponent_distance = float(np.linalg.norm(self.poses[opponent][:2] - self.poses[team][:2]))
        if block_gate > 0.72 and risk > 0.62 and opponent_distance < 1.05:
            return not self._near_own_critical_assets(team)
        return block_gate > 0.92 and risk > 0.86 and not self._near_own_critical_assets(team)
    def _block_goal(self, team: str, risk: float) -> np.ndarray:
        opponent = self._opponent(team)
        own_base = YELLOW_BASE_XY if team == "yellow" else BLUE_BASE_XY
        opponent_xy = self.poses[opponent][:2]
        toward_opponent = opponent_xy - own_base
        distance = float(np.linalg.norm(toward_opponent))
        if distance < 1e-6:
            toward_opponent = np.array([-1.0, 1.0], dtype=np.float32) if team == "yellow" else np.array([1.0, -1.0], dtype=np.float32)
            distance = float(np.linalg.norm(toward_opponent))
        unit = toward_opponent / max(distance, 1e-6)
        if risk > 0.82 and not self._near_own_critical_assets(team):
            return self._nearest_free_xy(opponent_xy)
        lane_distance = 0.54 + 0.26 * risk
        if risk > 0.72:
            lane_distance = min(distance - 0.08, 0.86)
        goal = own_base + unit * max(0.35, lane_distance)
        return self._nearest_free_xy(goal)
    def _route_distance_to(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> float:
        resolution = 0.05
        def to_key(point: np.ndarray) -> tuple[int, int]:
            return (int(round(float(point[0]) / resolution)), int(round(float(point[1]) / resolution)))
        obstacle_signature = tuple(
            sorted((round(float(v[0]) / resolution), round(float(v[1]) / resolution)) for v in self.pushable_obstacles.values())
        )
        cache_key = (to_key(start_xy), to_key(goal_xy), obstacle_signature)
        cached = self._route_distance_cache.get(cache_key)
        if cached is not None:
            return cached
        direct = float(np.linalg.norm(goal_xy - start_xy))
        if not self._segment_blocked_for_nav(start_xy, goal_xy):
            distance = direct
        else:
            path = self._astar_path(start_xy, goal_xy)
            if len(path) < 2:
                distance = math.inf
            else:
                distance = float(np.linalg.norm(path[0] - start_xy))
                for index in range(1, len(path)):
                    distance += float(np.linalg.norm(path[index] - path[index - 1]))
        self._route_distance_cache[cache_key] = distance
        if len(self._route_distance_cache) > 2048:
            self._route_distance_cache.clear()
        return distance
    def _nearest_free_xy(self, point: np.ndarray) -> np.ndarray:
        limit = HALF_ARENA - ROBOT_RADIUS - 0.02
        clamped = np.array(
            [
                float(np.clip(point[0], -limit, limit)),
                float(np.clip(point[1], -limit, limit)),
            ],
            dtype=np.float32,
        )
        if not self._pose_blocked(np.array([clamped[0], clamped[1], 0.0], dtype=np.float32)):
            return clamped
        for radius in (0.08, 0.14, 0.20, 0.28):
            for index in range(16):
                angle = math.tau * index / 16.0
                candidate = clamped + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * radius
                candidate[0] = float(np.clip(candidate[0], -limit, limit))
                candidate[1] = float(np.clip(candidate[1], -limit, limit))
                if not self._pose_blocked(np.array([candidate[0], candidate[1], 0.0], dtype=np.float32)):
                    return candidate
        return clamped
    def _pose_blocked(self, pose: np.ndarray) -> bool:
        return (
            self._static_pose_blocked(pose)
            or self._pushable_collision_name(pose) is not None
            or self._target_collision_name(pose) is not None
        )
    def _point_blocked_for_nav(self, x: float, y: float) -> bool:
        if abs(x) + ROBOT_RADIUS >= HALF_ARENA or abs(y) + ROBOT_RADIUS >= HALF_ARENA:
            return True
        eps = 1e-4
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        # Pushable boxes are rigid contacts, not static walls. The global route
        # planner may pass through them so the local integrator can solve a
        # persistent push instead of declaring the rest of the arena unreachable.
        for target in self.targets:
            if target.knocked:
                continue
            radius = BASE_TARGET_CONTACT_RADIUS if target.kind.startswith("base_") else NORMAL_TARGET_CONTACT_RADIUS
            dx = target.xy[0] - x
            dy = target.xy[1] - y
            if dx * dx + dy * dy <= (ROBOT_RADIUS + radius) ** 2:
                return True
        return False
    def _static_pose_blocked(self, pose: np.ndarray) -> bool:
        if self._footprint_outside_arena(pose):
            return True
        x, y = float(pose[0]), float(pose[1])
        eps = 1e-4
        for center, half_size in self.nav_blockers:
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=True):
            if abs(x - center[0]) < half_size[0] - eps and abs(y - center[1]) < half_size[1] - eps:
                return True
        for center, half_size in active_base_armor_blockers(self.armor, inflated=False):
            if self._pose_overlaps_aabb(pose, center, half_size, margin=0.010):
                return True
        return False
    def _footprint_outside_arena(self, pose: np.ndarray) -> bool:
        return self._arena_footprint_margin(pose) < 0.0
    def _arena_footprint_margin(self, pose: np.ndarray) -> float:
        yaw = float(pose[2])
        half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
        half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
        return min(HALF_ARENA - (abs(float(pose[0])) + half_x), HALF_ARENA - (abs(float(pose[1])) + half_y))
    def _pose_overlaps_aabb(
        self,
        pose: np.ndarray,
        center: tuple[float, float],
        half_size: tuple[float, float],
        *,
        margin: float = 0.0,
    ) -> bool:
        yaw = float(pose[2])
        half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
        half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
        return (
            abs(float(pose[0]) - center[0]) <= half_size[0] + half_x + margin
            and abs(float(pose[1]) - center[1]) <= half_size[1] + half_y + margin
        )
    def _boundary_escape_linear_speed(self, pose: np.ndarray) -> float:
        if self._arena_footprint_margin(pose) > 0.075 and self._local_blocker_cost(pose[:2]) < 0.55:
            return 0.0
        yaw = float(pose[2])
        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        inward = np.array(
            [
                -math.copysign(1.0, float(pose[0])) if abs(float(pose[0])) > 1.08 else 0.0,
                -math.copysign(1.0, float(pose[1])) if abs(float(pose[1])) > 1.08 else 0.0,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(inward))
        if norm <= 1e-6:
            return 0.0
        inward /= norm
        forward_score = float(np.dot(heading, inward))
        if abs(forward_score) < 0.18:
            return 0.0
        return 0.10 if forward_score > 0.0 else -0.10
