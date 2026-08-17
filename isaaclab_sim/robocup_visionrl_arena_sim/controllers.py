from __future__ import annotations

import math


from ._bootstrap import (
    ARENA_SIZE,
    BASE_ARMOR,
    BASE_RUSH_MIN_QUALITY,
    BASE_SHOOT_IDEAL_DISTANCE,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    BLOCK_HOLD_S,
    BLOCK_LATE_TIME_S,
    BLOCK_LEAD_SCORE,
    BLUE_DEMO_START_XY,
    BLUE_START_XY,
    DEMO_POLICY_TASKS,
    FIRE_COOLDOWN,
    LASER_DWELL_REQUIRED_S,
    LAST_FIRE_TIME,
    LINEAR_ACCEL_LIMIT,
    LOCALIZATION_CONTACT_LOSS,
    LOCALIZATION_RECOVERY_ROTATION_RAD,
    LOCALIZATION_RECOVERY_THRESHOLD,
    LOCALIZATION_SPIN_GAIN,
    LOCALIZATION_STUCK_LOSS,
    MATCH_AIM_TIME,
    MATCH_CONTROLLERS,
    MATCH_DRIVE_SPEED,
    MATCH_DURATION_S,
    MATCH_STATE,
    MATCH_TASKS,
    MAX_CONTACT_CORRECTION_STEP,
    MIN_TURN_ALIGNMENT,
    OPPONENT_AVOID_BEARING_RAD,
    OPPONENT_AVOID_RANGE,
    OPPONENT_THREAT_BLOCK_THRESHOLD,
    ROBOT_COLLISION_RADIUS,
    ROBOT_WIDTH,
    ROUTE_CLEARANCE,
    SHOOTER_FORWARD_OFFSET,
    SHOOT_IDEAL_DISTANCE,
    SHOOT_RANGE,
    TARGET_REGISTRY,
    WHEEL_ACCEL_LIMIT,
    WHEEL_RADIUS,
    WHEEL_SPEED_LIMIT,
    WHEEL_WIDTH,
    YELLOW_DEMO_START_XY,
    YELLOW_START_XY
)
from .costmap import (
    apply_costmap_recovery,
    costmap_potential,
    demo_policy_corridor,
    path_length,
    plan_safe_path,
    push_pushable_obstacle,
    slew_rate,
    warn_costmap,
    wrap_angle
)
from .laser import (
    apply_fire_rule,
    laser_accuracy_from_geometry,
    line_blocked_by_wall,
    reset_laser_lock,
    update_laser_lock
)
from .replay import point_blocked, pushable_collision_path, segment_blocked
from .rules import (
    base_attack_pose_quality,
    base_hit_success_cap,
    empty_opponent_estimate,
    normal_hits_against,
    opponent_bearing_estimate,
    opponent_team,
    shooting_range_limits,
    static_fire_pose,
    target_name_from_path,
    team_base_xy,
    team_score
)
from .spawn import target_path_from_name

class StrategyTeamController:
    def __init__(
        self,
        team: str,
        start_xy: tuple[float, float],
        start_yaw: float,
        tasks: list[tuple[str, tuple[float, float]]],
        speed: float,
    ):
        self.team = team
        self.pose = ((start_xy[0], start_xy[1], 0.0), start_yaw)
        self.tasks = tasks
        self.speed = speed
        self.task_index = 0
        self.state = "plan"
        self.path: list[tuple[float, float]] = []
        self.waypoint_index = 1
        self.last_update_t = 0.0
        self.aim_start_time = 0.0
        self.current_target_path = ""
        self.left_wheel_spin = 0.0
        self.right_wheel_spin = 0.0
        self.last_left_wheel_speed = 0.0
        self.last_right_wheel_speed = 0.0
        self.last_linear_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.motion_blocked = False
        self.blocked_since = 0.0
        self.recover_until = 0.0
        self.recover_spin_direction = 1.0
        self.last_progress_distance = float("inf")
        self.last_progress_t = 0.0
        self.current_fire_xy: tuple[float, float] | None = None
        self.block_until = 0.0
        self.last_strategy_print = -99.0
        self.localization_confidence = 1.0
        self.relocalize_rotation = 0.0
        self.last_contact_t = -99.0
        self.start_delay = 0.0
        self.opponent_estimate = empty_opponent_estimate()
        self.target_fail_counts: dict[str, int] = {}
        self.target_cooldowns: dict[str, float] = {}

    def set_pose(self, pose: tuple[tuple[float, float, float], float]):
        self.pose = pose

    def notify_contact(self, t: float):
        if t - self.last_contact_t < 0.35:
            return
        self.last_contact_t = t
        self.localization_confidence = max(0.05, self.localization_confidence - LOCALIZATION_CONTACT_LOSS * 0.25)
        self.recover_until = t + 0.55
        self.recover_spin_direction = -1.0 if self.team == "yellow" else 1.0
        if self.state != "relocalize":
            self.state = "recover"
        self._print_strategy(t, f"collision contact; backing off before replanning confidence={self.localization_confidence:.2f}")

    def notify_robot_contact(self, t: float):
        if t - self.last_contact_t < 0.35:
            return
        self.last_contact_t = t
        self._print_strategy(t, "tactical robot contact; opponent pose known, localization unchanged")

    def update(self, t: float) -> tuple[tuple[float, float, float], float]:
        dt = max(0.0, min(0.05, t - self.last_update_t)) if self.last_update_t > 0.0 else 0.0
        self.last_update_t = t
        self.opponent_estimate = self._estimate_opponent()
        if MATCH_STATE["winner"] is not None:
            self.last_linear_velocity = 0.0
            self.last_angular_velocity = 0.0
            return self.pose

        if t < self.start_delay:
            self._integrate_differential(0.0, 0.0, dt)
            return self.pose

        if self.state == "relocalize":
            self._spin_relocalize(dt)
            if self.relocalize_rotation >= LOCALIZATION_RECOVERY_ROTATION_RAD:
                self.localization_confidence = 1.0
                self.state = "plan"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
                self._print_strategy(t, "localization rebuilt from lidar/imu/camera scan")
            return self.pose

        if self.state == "recover":
            self._integrate_differential(-0.070, self.recover_spin_direction * 0.32, dt)
            if t >= self.recover_until:
                self.state = "plan"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
            return self.pose

        if self.state == "plan":
            if self.localization_confidence < LOCALIZATION_RECOVERY_THRESHOLD:
                self.state = "relocalize"
                self.relocalize_rotation = 0.0
                self._print_strategy(t, f"localization confidence={self.localization_confidence:.2f}; spinning before next decision")
                return self.pose
            strategy = self._select_strategy(t)
            if strategy["mode"] == "wait":
                self._integrate_differential(0.0, 0.0, dt)
                return self.pose

            if strategy["mode"] == "block":
                block_xy = strategy["fire_xy"]
                assert isinstance(block_xy, tuple)
                start_xy = (self.pose[0][0], self.pose[0][1])
                self.path = self._plan_path(start_xy, block_xy)
                self.waypoint_index = 1
                self.current_target_path = ""
                self.current_fire_xy = block_xy
                self.block_until = t + BLOCK_HOLD_S
                self.state = "drive_block"
                self.blocked_since = 0.0
                self.last_progress_distance = float("inf")
                self.last_progress_t = t
                self._print_strategy(t, f"blocking central lane at ({block_xy[0]:.2f}, {block_xy[1]:.2f})")
                return self.pose

            target_path = str(strategy["target_path"])
            fire_xy = strategy["fire_xy"]
            assert isinstance(fire_xy, tuple)
            self.current_target_path = target_path
            self.current_fire_xy = fire_xy
            start_xy = (self.pose[0][0], self.pose[0][1])
            self.path = self._plan_path(start_xy, fire_xy)
            self.waypoint_index = 1
            self.state = "drive"
            self.blocked_since = 0.0
            self.last_progress_distance = float("inf")
            self.last_progress_t = t
            target_name = target_name_from_path(target_path)
            self._print_strategy(t, f"attacking {target_name} from ({fire_xy[0]:.2f}, {fire_xy[1]:.2f})")

        if self.state in ("drive", "drive_block"):
            arrived = self._drive_differential(dt)
            if arrived and self.state == "drive":
                self.aim_start_time = t
                self.state = "aim"
            elif arrived and self.state == "drive_block":
                self.state = "block"
            elif self._drive_is_stuck(t):
                self.recover_until = t + 0.80
                self.recover_spin_direction = -1.0 if self.team == "yellow" else 1.0
                self.state = "recover"
                self.localization_confidence = max(0.08, self.localization_confidence - LOCALIZATION_STUCK_LOSS)
                print(f"[MATCH]: {self.team} blocked; backing off and replanning.")

        if self.state == "block":
            opponent_pose = self._opponent_pose()
            if opponent_pose is not None:
                opponent_xy = (opponent_pose[0][0], opponent_pose[0][1])
                self._aim_differential(opponent_xy, dt)
            else:
                self._integrate_differential(0.0, 0.35 if self.team == "yellow" else -0.35, dt)
            if t >= self.block_until or not self._should_block(t):
                self.state = "plan"

        if self.state == "aim":
            target = TARGET_REGISTRY[self.current_target_path]
            target_xy = target["xy"]
            assert isinstance(target_xy, tuple)
            aligned = self._aim_differential(target_xy, dt)
            if aligned and t - self.aim_start_time >= MATCH_AIM_TIME and t - LAST_FIRE_TIME[self.team] >= FIRE_COOLDOWN:
                hit_path = update_laser_lock(self.team, self.pose, t)
                if hit_path == self.current_target_path:
                    LAST_FIRE_TIME[self.team] = t
                    knocked = apply_fire_rule(self.team, self.current_target_path)
                    reset_laser_lock(self.team)
                    if knocked or target["knocked"]:
                        self.task_index += 1
                        self.state = "plan"
                elif t - self.aim_start_time >= MATCH_AIM_TIME + LASER_DWELL_REQUIRED_S + 0.70:
                    reset_laser_lock(self.team)
                    self._mark_target_failed(self.current_target_path, t, "no clean laser dwell")
                    self.state = "plan"
            elif not aligned:
                reset_laser_lock(self.team)

        return self.pose

    def _select_strategy(self, t: float) -> dict[str, object]:
        if self._should_block(t):
            block_xy = self._select_block_point()
            if block_xy is not None:
                return {"mode": "block", "fire_xy": block_xy}

        candidates = self._attack_candidates(t)
        if not candidates:
            return {"mode": "wait"}

        scored = [(self._score_attack(candidate, t), candidate) for candidate in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        if best_score <= -50.0:
            return {"mode": "wait"}
        best_candidate["mode"] = "attack"
        return best_candidate

    def _plan_path(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
        return plan_safe_path(start_xy, goal_xy)

    def _attack_candidates(self, t: float) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        start_xy = (self.pose[0][0], self.pose[0][1])
        for target_path, target in TARGET_REGISTRY.items():
            if target["knocked"] or target["owner"] == self.team:
                continue
            if self._target_on_cooldown(target_path, t):
                continue
            kind = str(target["kind"])
            if kind != "normal" and kind != f"base_{opponent_team(self.team)}":
                continue
            solution = self._best_fire_solution(start_xy, target_path)
            if solution is None:
                continue
            fire_xy, route_len, shot_quality = solution
            candidates.append(
                {
                    "target_path": target_path,
                    "fire_xy": fire_xy,
                    "route_len": route_len,
                    "shot_quality": shot_quality,
                    "kind": kind,
                }
            )
        return candidates

    def _target_on_cooldown(self, target_path: str, t: float) -> bool:
        return t < self.target_cooldowns.get(target_path, -99.0)

    def _mark_target_failed(self, target_path: str, t: float, reason: str):
        count = self.target_fail_counts.get(target_path, 0) + 1
        self.target_fail_counts[target_path] = count
        cooldown = 4.0 + 3.0 * min(count, 3)
        self.target_cooldowns[target_path] = t + cooldown
        target_name = target_name_from_path(target_path)
        self._print_strategy(t, f"shot withheld on {target_name}; {reason}, cooldown {cooldown:.1f}s")

    def _best_fire_solution(
        self,
        start_xy: tuple[float, float],
        target_path: str,
    ) -> tuple[tuple[float, float], float, float] | None:
        target = TARGET_REGISTRY[target_path]
        target_xy = target["xy"]
        assert isinstance(target_xy, tuple)
        yaw = float(target["yaw"])
        kind = str(target["kind"])
        target_name = target_name_from_path(target_path)
        front = (math.cos(yaw), math.sin(yaw))
        tangent = (-front[1], front[0])
        fire_candidates: list[tuple[float, float]] = []
        fixed = static_fire_pose(self.team, target_name, self.tasks)
        if fixed is not None:
            fire_candidates.append(fixed)
        if kind.startswith("base_"):
            base_xy = team_base_xy(str(target["owner"]))
            hits = max(1, min(4, normal_hits_against(self.team)))
            if kind == "base_blue":
                opened_dirs = (
                    (1.0, 0.0),
                    (0.0, -1.0),
                    (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
                )
            else:
                opened_dirs = (
                    (-1.0, 0.0),
                    (0.0, 1.0),
                    (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
                )
            allowed_dirs = opened_dirs[:1] if hits == 1 else opened_dirs[:2] if hits == 2 else opened_dirs
            for direction in allowed_dirs:
                side_tangent = (-direction[1], direction[0])
                for radius in (0.48, 0.62, 0.78, 0.94):
                    for lateral in (-0.10, -0.04, 0.0, 0.04, 0.10):
                        fire_candidates.append(
                            (
                                base_xy[0] + direction[0] * radius + side_tangent[0] * lateral,
                                base_xy[1] + direction[1] * radius + side_tangent[1] * lateral,
                            )
                        )
            wall_limit = ARENA_SIZE * 0.5 - ROBOT_COLLISION_RADIUS - 0.045
            for offset in (-0.16, -0.08, 0.0, 0.08, 0.16):
                if kind == "base_blue":
                    fire_candidates.append((-wall_limit, target_xy[1] + offset))
                    fire_candidates.append((target_xy[0] + offset, wall_limit))
                else:
                    fire_candidates.append((wall_limit, target_xy[1] + offset))
                    fire_candidates.append((target_xy[0] + offset, -wall_limit))
            angle_offsets = {
                1: (-1.12, 1.12, -0.86, 0.86),
                2: (-0.96, 0.96, -0.70, 0.70),
                3: (-0.78, 0.78, -0.44, 0.44, -0.26, 0.26),
                4: (-0.62, 0.62, -0.34, 0.34, -0.16, 0.16, 0.0),
            }[hits]
            for distance in (
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_MIN_RANGE + 0.02,
                SHOOTER_FORWARD_OFFSET + 0.34,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_IDEAL_DISTANCE,
                SHOOTER_FORWARD_OFFSET + BASE_SHOOT_RANGE - 0.04,
            ):
                for angle_offset in angle_offsets:
                    direction = (
                        math.cos(yaw + angle_offset),
                        math.sin(yaw + angle_offset),
                    )
                    fire_candidates.append(
                        (
                            target_xy[0] + direction[0] * distance,
                            target_xy[1] + direction[1] * distance,
                        )
                    )
        for distance in (
            SHOOTER_FORWARD_OFFSET + (BASE_SHOOT_MIN_RANGE + 0.02 if kind.startswith("base_") else 0.14),
            SHOOTER_FORWARD_OFFSET + (0.36 if kind.startswith("base_") else 0.24),
            SHOOTER_FORWARD_OFFSET + (BASE_SHOOT_IDEAL_DISTANCE if kind.startswith("base_") else SHOOT_IDEAL_DISTANCE),
            SHOOTER_FORWARD_OFFSET + ((BASE_SHOOT_RANGE if kind.startswith("base_") else SHOOT_RANGE) - 0.03),
        ):
            lateral_offsets = (0.0, -0.08, 0.08, -0.14, 0.14)
            if kind.startswith("base_"):
                lateral_offsets = (0.0, -0.10, 0.10, -0.18, 0.18)
            for side_offset in lateral_offsets:
                fire_candidates.append(
                    (
                        target_xy[0] + front[0] * distance + tangent[0] * side_offset,
                        target_xy[1] + front[1] * distance + tangent[1] * side_offset,
                    )
                )

        best: tuple[tuple[float, float], float, float] | None = None
        best_score = -999.0
        seen: set[tuple[int, int]] = set()
        for fire_xy in fire_candidates:
            key = (round(fire_xy[0] * 100), round(fire_xy[1] * 100))
            if key in seen:
                continue
            seen.add(key)
            if self._fire_pose_rejected(fire_xy, target_xy, kind.startswith("base_")):
                continue
            if kind.startswith("base_"):
                pose_quality = base_attack_pose_quality(self.team, target_path, fire_xy)
                if pose_quality <= 0.0:
                    continue
            try:
                route = plan_safe_path(start_xy, fire_xy)
            except RuntimeError:
                continue
            route_len = path_length(route)
            center_distance = math.hypot(target_xy[0] - fire_xy[0], target_xy[1] - fire_xy[1])
            shot_distance = max(0.0, center_distance - SHOOTER_FORWARD_OFFSET)
            quality = laser_accuracy_from_geometry(shot_distance, 0.0, kind.startswith("base_"))
            if kind.startswith("base_"):
                quality = min(base_hit_success_cap(self.team), quality * base_attack_pose_quality(self.team, target_path, fire_xy))
            quality += 0.15 if not line_blocked_by_wall(fire_xy, target_xy) else -0.45
            score = quality - route_len * 0.10 - costmap_potential(fire_xy) * 0.18
            if score > best_score:
                best_score = score
                best = (fire_xy, route_len, max(0.0, min(1.0, quality)))
        return best

    def _fire_pose_rejected(self, fire_xy: tuple[float, float], target_xy: tuple[float, float], base_target: bool) -> bool:
        if not (-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE <= fire_xy[0] <= ARENA_SIZE * 0.5 - ROUTE_CLEARANCE):
            return True
        if not (-ARENA_SIZE * 0.5 + ROUTE_CLEARANCE <= fire_xy[1] <= ARENA_SIZE * 0.5 - ROUTE_CLEARANCE):
            return True
        if point_blocked(fire_xy):
            return True
        if costmap_potential(fire_xy) > 3.0:
            return True
        center_distance = math.hypot(target_xy[0] - fire_xy[0], target_xy[1] - fire_xy[1])
        shot_distance = max(0.0, center_distance - SHOOTER_FORWARD_OFFSET)
        min_range, max_range = shooting_range_limits(base_target)
        if shot_distance < min_range or shot_distance > max_range:
            return True
        if line_blocked_by_wall(fire_xy, target_xy):
            return True
        return False

    def _score_attack(self, candidate: dict[str, object], t: float) -> float:
        opponent = opponent_team(self.team)
        own_score = team_score(self.team)
        opponent_score = team_score(opponent)
        score_delta = own_score - opponent_score
        time_remaining = max(0.0, MATCH_DURATION_S - t)
        route_len = float(candidate["route_len"])
        shot_quality = float(candidate["shot_quality"])
        kind = str(candidate["kind"])
        aggression = 0.38
        if score_delta < 0:
            aggression += 0.28
        if time_remaining < 80.0:
            aggression += 0.22
        if time_remaining < 35.0:
            aggression += 0.22
        if len(BASE_ARMOR[self.team]) <= 2:
            aggression += 0.12

        if kind == f"base_{opponent}":
            if shot_quality < BASE_RUSH_MIN_QUALITY:
                return -60.0
            base_rush_risk = 3.0 * float(self.opponent_estimate["threat_to_own_base"]) if score_delta >= 0 else 0.0
            return 21.0 + 42.0 * shot_quality + 18.0 * aggression - 1.2 * route_len - base_rush_risk

        defense_risk = float(self.opponent_estimate["threat_to_own_base"]) * (7.0 if score_delta >= 0 else 3.0)
        return 5.0 + 8.0 * shot_quality - 1.5 * route_len + max(0.0, -score_delta) * 0.10 - defense_risk

    def _should_block(self, t: float) -> bool:
        opponent = opponent_team(self.team)
        time_remaining = MATCH_DURATION_S - t
        score_delta = team_score(self.team) - team_score(opponent)
        if score_delta >= BLOCK_LEAD_SCORE and time_remaining <= BLOCK_LATE_TIME_S:
            return True
        estimate = self.opponent_estimate
        if not estimate["available"] or score_delta < 5:
            return False
        threat = float(estimate["threat_to_own_base"])
        if threat >= OPPONENT_THREAT_BLOCK_THRESHOLD:
            return True
        return bool(estimate["visible"]) and float(estimate["distance_to_own_base"]) < 0.90

    def _select_block_point(self) -> tuple[float, float] | None:
        opponent_pose = self._opponent_pose()
        our_base = team_base_xy(self.team)
        if opponent_pose is not None and self.opponent_estimate["available"]:
            dx = float(self.opponent_estimate["dx"]) + self.pose[0][0] - our_base[0]
            dy = float(self.opponent_estimate["dy"]) + self.pose[0][1] - our_base[1]
            distance = max(1e-6, math.hypot(dx, dy))
            candidate = (our_base[0] + dx / distance * 0.72, our_base[1] + dy / distance * 0.72)
            if not point_blocked(candidate):
                return candidate

        fallback = (0.18, -0.18) if self.team == "yellow" else (-0.18, 0.18)
        if not point_blocked(fallback):
            return fallback
        return None

    def _opponent_pose(self) -> tuple[tuple[float, float, float], float] | None:
        controller = MATCH_CONTROLLERS.get(opponent_team(self.team))
        if controller is None:
            return None
        return controller.pose

    def _estimate_opponent(self) -> dict[str, float | bool]:
        opponent_pose = self._opponent_pose()
        if opponent_pose is None:
            return empty_opponent_estimate()
        return opponent_bearing_estimate(self.team, self.pose, opponent_pose)

    def _print_strategy(self, t: float, message: str):
        if t - self.last_strategy_print < 0.45:
            return
        self.last_strategy_print = t
        print(f"[STRATEGY]: {self.team} {message}.")

    def _spin_relocalize(self, dt: float):
        spin_direction = 1.0 if self.team == "yellow" else -1.0
        angular_velocity = spin_direction * 1.05
        self._integrate_differential(0.0, angular_velocity, dt)
        self.relocalize_rotation += abs(angular_velocity) * max(0.0, dt)
        self.localization_confidence = min(
            1.0,
            self.localization_confidence + LOCALIZATION_SPIN_GAIN * max(0.0, dt),
        )

    def _integrate_differential(self, linear_velocity: float, angular_velocity: float, dt: float):
        pos, yaw = self.pose
        if dt <= 0.0:
            self.last_linear_velocity = linear_velocity
            self.last_angular_velocity = angular_velocity
            return

        linear_velocity = max(-self.speed, min(self.speed, linear_velocity))
        angular_velocity = max(-2.4, min(2.4, angular_velocity))
        track_width = ROBOT_WIDTH + WHEEL_WIDTH
        desired_left_speed = linear_velocity - angular_velocity * track_width * 0.5
        desired_right_speed = linear_velocity + angular_velocity * track_width * 0.5
        desired_left_speed = max(-WHEEL_SPEED_LIMIT, min(WHEEL_SPEED_LIMIT, desired_left_speed))
        desired_right_speed = max(-WHEEL_SPEED_LIMIT, min(WHEEL_SPEED_LIMIT, desired_right_speed))
        left_speed = slew_rate(
            self.last_left_wheel_speed,
            desired_left_speed,
            min(LINEAR_ACCEL_LIMIT, WHEEL_ACCEL_LIMIT) * dt,
        )
        right_speed = slew_rate(
            self.last_right_wheel_speed,
            desired_right_speed,
            min(LINEAR_ACCEL_LIMIT, WHEEL_ACCEL_LIMIT) * dt,
        )
        linear_velocity = (left_speed + right_speed) * 0.5
        angular_velocity = (right_speed - left_speed) / track_width

        new_yaw = wrap_angle(yaw + angular_velocity * dt)
        mid_yaw = wrap_angle(yaw + angular_velocity * dt * 0.5)
        candidate = (
            pos[0] + linear_velocity * math.cos(mid_yaw) * dt,
            pos[1] + linear_velocity * math.sin(mid_yaw) * dt,
            0.0,
        )
        self.motion_blocked = False
        if segment_blocked((pos[0], pos[1]), (candidate[0], candidate[1])):
            warn_costmap(self.team, "collision sweep detected; holding pose and replanning")
            candidate = (pos[0], pos[1], 0.0)
            linear_velocity = 0.0
            left_speed = 0.0
            right_speed = 0.0
            self.motion_blocked = True

        pushable_path = None if self.motion_blocked else pushable_collision_path((candidate[0], candidate[1]))
        if pushable_path is not None:
            if linear_velocity > 0.035 and push_pushable_obstacle(pushable_path, mid_yaw, self.team):
                if pushable_collision_path((candidate[0], candidate[1])) is not None:
                    candidate = (pos[0], pos[1], 0.0)
                    linear_velocity = 0.0
                    left_speed = 0.0
                    right_speed = 0.0
                    self.motion_blocked = True
                else:
                    linear_velocity *= 0.62
                    left_speed *= 0.62
                    right_speed *= 0.62
            else:
                warn_costmap(self.team, f"pushable box {pushable_path.rsplit('/', 1)[-1]} cannot move; backing off")
                candidate = (pos[0], pos[1], 0.0)
                linear_velocity = 0.0
                left_speed = 0.0
                right_speed = 0.0
                self.motion_blocked = True

        corrected_xy, costmap_touch, hard_costmap_touch = apply_costmap_recovery((candidate[0], candidate[1]), self.team)
        if costmap_touch:
            correction_dx = corrected_xy[0] - candidate[0]
            correction_dy = corrected_xy[1] - candidate[1]
            correction_len = math.hypot(correction_dx, correction_dy)
            if correction_len > MAX_CONTACT_CORRECTION_STEP:
                scale = MAX_CONTACT_CORRECTION_STEP / correction_len
                corrected_xy = (candidate[0] + correction_dx * scale, candidate[1] + correction_dy * scale)
            candidate = (corrected_xy[0], corrected_xy[1], 0.0)
            self.motion_blocked = self.motion_blocked or hard_costmap_touch
            linear_velocity *= 0.35
            left_speed *= 0.35
            right_speed *= 0.35
        elif point_blocked((candidate[0], candidate[1])):
            safe_xy, _touched, _hard = apply_costmap_recovery((candidate[0], candidate[1]), self.team, passes=6)
            candidate = (safe_xy[0], safe_xy[1], 0.0)
            self.motion_blocked = True

        self.left_wheel_spin += left_speed * dt / WHEEL_RADIUS
        self.right_wheel_spin += right_speed * dt / WHEEL_RADIUS
        self.last_left_wheel_speed = left_speed
        self.last_right_wheel_speed = right_speed
        self.last_linear_velocity = linear_velocity
        self.last_angular_velocity = angular_velocity
        self.pose = (candidate, new_yaw)

    def _drive_is_stuck(self, t: float) -> bool:
        if self.motion_blocked:
            if self.blocked_since <= 0.0:
                self.blocked_since = t
            return t - self.blocked_since > 0.45
        self.blocked_since = 0.0

        if not self.path or self.waypoint_index >= len(self.path):
            return False
        pos, _ = self.pose
        waypoint = self.path[self.waypoint_index]
        distance = math.hypot(waypoint[0] - pos[0], waypoint[1] - pos[1])
        if distance < self.last_progress_distance - 0.018:
            self.last_progress_distance = distance
            self.last_progress_t = t
            return False
        if abs(self.last_linear_velocity) > 0.035 or abs(self.last_angular_velocity) > 0.20:
            return False
        if self.last_progress_t <= 0.0:
            self.last_progress_t = t
            self.last_progress_distance = distance
            return False
        return t - self.last_progress_t > 2.2

    def _drive_differential(self, dt: float) -> bool:
        if not self.path or self.waypoint_index >= len(self.path):
            self._integrate_differential(0.0, 0.0, dt)
            return True

        pos, yaw = self.pose
        while self.waypoint_index < len(self.path):
            waypoint = self.path[self.waypoint_index]
            if math.hypot(waypoint[0] - pos[0], waypoint[1] - pos[1]) > 0.075:
                break
            self.waypoint_index += 1

        if self.waypoint_index >= len(self.path):
            self._integrate_differential(0.0, 0.0, dt)
            return True

        waypoint = self.path[self.waypoint_index]
        desired_yaw = math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        heading_error = wrap_angle(desired_yaw - yaw)
        angular_velocity = max(-2.4, min(2.4, 3.2 * heading_error))
        alignment = max(0.0, 1.0 - abs(heading_error) / 1.20)
        linear_velocity = self.speed * max(MIN_TURN_ALIGNMENT, alignment)
        if abs(heading_error) > 1.35:
            linear_velocity = 0.0
        estimate = self.opponent_estimate
        if (
            estimate["available"]
            and estimate["visible"]
            and float(estimate["distance"]) < OPPONENT_AVOID_RANGE
            and abs(float(estimate["relative_bearing"])) < OPPONENT_AVOID_BEARING_RAD
        ):
            avoid_turn = -1.0 if float(estimate["relative_bearing"]) >= 0.0 else 1.0
            linear_velocity = min(linear_velocity, self.speed * 0.18)
            angular_velocity = max(-2.4, min(2.4, angular_velocity + avoid_turn * 0.75))
        self._integrate_differential(linear_velocity, angular_velocity, dt)
        return False

    def _aim_differential(self, target_xy: tuple[float, float], dt: float) -> bool:
        pos, yaw = self.pose
        desired_yaw = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0])
        heading_error = wrap_angle(desired_yaw - yaw)
        angular_velocity = max(-1.8, min(1.8, 3.8 * heading_error))
        if abs(heading_error) < math.radians(1.5):
            angular_velocity = 0.0
        self._integrate_differential(0.0, angular_velocity, dt)
        return abs(heading_error) < math.radians(4.0)


class PolicyReplayController(StrategyTeamController):
    """Motor-level replay of the learned high-level policy.

    The policy layer selects the next opponent target from a self-play style
    tactical sequence. The inherited controller still performs differential
    drive tracking, acceleration limiting, costmap avoidance, aiming, and
    shooter gating.
    """

    def _plan_path(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
        staged = demo_policy_corridor(self.team, start_xy, goal_xy)
        path: list[tuple[float, float]] = [start_xy]
        cursor = start_xy
        for waypoint in staged[1:]:
            segment = plan_safe_path(cursor, waypoint)
            path.extend(segment[1:])
            cursor = waypoint
        return path

    def _select_strategy(self, t: float) -> dict[str, object]:
        start_xy = (self.pose[0][0], self.pose[0][1])
        first_deferred_base = ""
        cursor = self.task_index
        while cursor < len(self.tasks):
            target_name, _nominal_fire_xy = self.tasks[cursor]
            target_path = target_path_from_name(target_name)
            target = TARGET_REGISTRY.get(target_path)
            if target is None:
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue
            if self._target_on_cooldown(target_path, t):
                if cursor == self.task_index and self.target_fail_counts.get(target_path, 0) >= 2:
                    self.task_index += 1
                cursor += 1
                continue
            if target["knocked"]:
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue

            kind = str(target["kind"])
            opponent = opponent_team(self.team)
            if target["owner"] == self.team:
                warn_costmap(self.team, f"policy rejected illegal own target {target_name}")
                if cursor == self.task_index:
                    self.task_index += 1
                cursor += 1
                continue
            if kind == f"base_{opponent}" and len(BASE_ARMOR[opponent]) > 3:
                first_deferred_base = target_name
                cursor += 1
                continue

            solution = self._best_fire_solution(start_xy, target_path)
            if solution is None:
                warn_costmap(self.team, f"policy cannot find clean fire pose for {target_name}; trying next tactical option")
                cursor += 1
                continue

            fire_xy, route_len, shot_quality = solution
            if cursor != self.task_index:
                self._print_strategy(t, f"policy skipped {cursor - self.task_index} blocked option(s)")
            self.task_index = cursor
            self._print_strategy(
                t,
                f"policy action target={target_name} fire_pose=({fire_xy[0]:.2f}, {fire_xy[1]:.2f})",
            )
            return {
                "mode": "attack",
                "target_path": target_path,
                "fire_xy": fire_xy,
                "route_len": route_len,
                "shot_quality": shot_quality,
                "kind": kind,
            }
        if first_deferred_base:
            warn_costmap(self.team, f"base target {first_deferred_base} deferred until opponent armor is removed")
        return {"mode": "wait"}

    def _should_block(self, t: float) -> bool:
        return False


def initialize_match_controllers():
    if MATCH_CONTROLLERS:
        return
    for team, task_specs in MATCH_TASKS.items():
        for target_name, fire_xy in task_specs:
            target_path = target_path_from_name(target_name)
            if target_path not in TARGET_REGISTRY:
                raise RuntimeError(f"Match task target not found: {target_path}")
            owner = TARGET_REGISTRY[target_path]["owner"]
            if owner == team:
                raise RuntimeError(f"{team} task illegally targets its own target: {target_name}")
            if point_blocked(fire_xy):
                raise RuntimeError(f"Match task fire pose is blocked: {team} {target_name} {fire_xy}")

    MATCH_CONTROLLERS["yellow"] = StrategyTeamController(
        "yellow",
        YELLOW_START_XY,
        math.pi * 0.5,
        MATCH_TASKS["yellow"],
        MATCH_DRIVE_SPEED,
    )
    MATCH_CONTROLLERS["blue"] = StrategyTeamController(
        "blue",
        BLUE_START_XY,
        -math.pi * 0.5,
        MATCH_TASKS["blue"],
        MATCH_DRIVE_SPEED * 0.98,
    )


def initialize_demo_flow_controllers():
    if MATCH_CONTROLLERS:
        return
    for team, task_specs in DEMO_POLICY_TASKS.items():
        for target_name, fire_xy in task_specs:
            target_path = target_path_from_name(target_name)
            if target_path not in TARGET_REGISTRY:
                raise RuntimeError(f"Match task target not found: {target_path}")
            owner = TARGET_REGISTRY[target_path]["owner"]
            if owner == team:
                raise RuntimeError(f"{team} task illegally targets its own target: {target_name}")
            if point_blocked(fire_xy):
                warn_costmap("policy", f"nominal fire pose for {team}/{target_name} is occupied; live planner will choose another pose")

    MATCH_CONTROLLERS["yellow"] = PolicyReplayController(
        "yellow",
        YELLOW_DEMO_START_XY,
        math.pi * 0.5,
        DEMO_POLICY_TASKS["yellow"],
        MATCH_DRIVE_SPEED * 0.58,
    )
    MATCH_CONTROLLERS["blue"] = PolicyReplayController(
        "blue",
        BLUE_DEMO_START_XY,
        -math.pi * 0.5,
        DEMO_POLICY_TASKS["blue"],
        MATCH_DRIVE_SPEED * 0.56,
    )
