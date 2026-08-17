from __future__ import annotations

import math
from pathlib import Path


from ._bootstrap import (
    ARMOR_REMOVALS,
    BASE_ARMOR,
    BASE_ARMOR_LIFT_CLEARANCE_Z,
    BASE_HIT_RADIUS,
    BASE_SHOOT_RANGE,
    BASE_TARGET_CONTACT_RADIUS,
    FIRE_COOLDOWN,
    LASER_BLOCKERS,
    LASER_DWELL_FULL_CONFIDENCE_S,
    LASER_DWELL_REQUIRED_S,
    LASER_LOCKS,
    LAST_FIRE_TIME,
    MATCH_CONTROLLERS,
    MATCH_STATE,
    PUSHABLE_OBSTACLES,
    ROBOT_COLLISION_RADIUS,
    SHOOTER_POSE,
    SHOOT_HIT_RADIUS,
    SHOOT_RANGE,
    TARGET_CONTACT_RADIUS,
    TARGET_FALLS,
    TARGET_REGISTRY,
    args_cli,
    get_current_stage
)
from .rules import (
    base_attack_pose_quality,
    base_hit_success_cap,
    normal_hits_against,
    normalized_laser_dwell_factor,
    shooting_range_limits,
    target_name_from_path
)
from .spawn import segment_intersects_aabb, unregister_blocker
from .transforms import (
    get_xform,
    local_to_world,
    quat_from_euler,
    set_visibility,
    set_xform
)

def line_blocked_by_wall(origin_xy: tuple[float, float], target_xy: tuple[float, float]) -> bool:
    for blocker_path, center, half_size in LASER_BLOCKERS:
        if segment_intersects_aabb(origin_xy, target_xy, center, half_size):
            return True
    for obstacle in PUSHABLE_OBSTACLES.values():
        xy = obstacle["xy"]
        half = obstacle["half"]
        assert isinstance(xy, list)
        assert isinstance(half, tuple)
        if segment_intersects_aabb(origin_xy, target_xy, (float(xy[0]), float(xy[1])), half):
            return True
    return False


def laser_accuracy_from_geometry(distance: float, lateral_error: float, base_target: bool) -> float:
    min_range, max_range = shooting_range_limits(base_target)
    if distance < min_range or distance > max_range:
        return 0.0
    hit_radius = BASE_HIT_RADIUS if base_target else SHOOT_HIT_RADIUS
    if lateral_error > hit_radius:
        return 0.0
    distance_quality = (max_range - distance) / max(1e-6, max_range - min_range)
    lateral_quality = 1.0 - lateral_error / max(hit_radius, 1e-6)
    accuracy = 0.18 + 0.64 * distance_quality + 0.18 * lateral_quality
    if base_target:
        accuracy -= 0.10
    return max(0.05, min(0.98, accuracy))


def deterministic_laser_draw(team: str, target_path: str, distance: float, dwell_s: float = 0.0) -> float:
    phase = float(MATCH_STATE["current_time"]) * 7.131 + len(target_path) * 0.173 + (0.37 if team == "yellow" else 0.71)
    phase += distance * 11.0 + dwell_s * 2.337
    return abs(math.sin(phase) * 43758.5453) % 1.0


def reset_laser_lock(team: str):
    LASER_LOCKS[team]["target_path"] = ""
    LASER_LOCKS[team]["start_time"] = -99.0


def detect_laser_candidate(
    team: str,
    pose: tuple[tuple[float, float, float], float],
) -> tuple[str, float, float, float] | None:
    robot_pos, yaw = pose
    shooter_origin = local_to_world(robot_pos, SHOOTER_POSE, 0.0, 0.0, yaw)
    origin_xy = (shooter_origin[0], shooter_origin[1])
    forward = (math.cos(yaw), math.sin(yaw))
    best_path = None
    best_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0
    best_accuracy = 0.0
    best_lateral = 0.0
    own_candidate_projection = max(SHOOT_RANGE, BASE_SHOOT_RANGE) + 1.0

    for target_path, target in TARGET_REGISTRY.items():
        if target["knocked"]:
            continue
        kind = str(target["kind"])
        owner = str(target["owner"])
        if kind.startswith("base_") and owner != team and normal_hits_against(team) <= 0:
            continue
        target_xy = target["xy"]
        assert isinstance(target_xy, tuple)
        dx = target_xy[0] - origin_xy[0]
        dy = target_xy[1] - origin_xy[1]
        projection = dx * forward[0] + dy * forward[1]
        min_range, max_range = shooting_range_limits(kind.startswith("base_"))
        if projection < min_range or projection > max_range:
            continue
        perpendicular = abs(dx * forward[1] - dy * forward[0])
        hit_radius = BASE_HIT_RADIUS if kind.startswith("base_") else SHOOT_HIT_RADIUS
        if perpendicular > hit_radius:
            continue
        if line_blocked_by_wall(origin_xy, target_xy):
            continue
        if owner == team:
            own_candidate_projection = min(own_candidate_projection, projection)
            continue
        accuracy = laser_accuracy_from_geometry(projection, perpendicular, kind.startswith("base_"))
        if kind.startswith("base_"):
            pose_quality = base_attack_pose_quality(team, target_path, (robot_pos[0], robot_pos[1]))
            if pose_quality <= 0.0:
                continue
            accuracy = min(base_hit_success_cap(team), accuracy * pose_quality)
        if projection < best_projection:
            best_projection = projection
            best_path = target_path
            best_accuracy = accuracy
            best_lateral = perpendicular
    if own_candidate_projection <= best_projection:
        reset_laser_lock(team)
        MATCH_STATE["last_event"] = f"{team} own-target safety gate blocked laser"
        return None
    if best_path is None:
        reset_laser_lock(team)
        return None
    return best_path, best_projection, best_lateral, best_accuracy


def detect_laser_hit(team: str, pose: tuple[tuple[float, float, float], float]) -> str | None:
    candidate = detect_laser_candidate(team, pose)
    if candidate is None:
        return None
    best_path, best_projection, best_lateral, best_accuracy = candidate
    draw = deterministic_laser_draw(team, best_path, best_projection, LASER_DWELL_FULL_CONFIDENCE_S)
    if draw > best_accuracy:
        MATCH_STATE["last_event"] = (
            f"{team} laser miss: d={best_projection:.2f}m lateral={best_lateral:.3f} "
            f"p={best_accuracy:.2f}"
        )
        return None
    return best_path


def update_laser_lock(team: str, pose: tuple[tuple[float, float, float], float], t: float) -> str | None:
    candidate = detect_laser_candidate(team, pose)
    if candidate is None:
        return None
    target_path, distance, lateral, accuracy = candidate
    lock = LASER_LOCKS[team]
    if lock["target_path"] != target_path:
        lock["target_path"] = target_path
        lock["start_time"] = t
        MATCH_STATE["last_event"] = f"{team} laser locked {target_name_from_path(target_path)}; dwell 0.00/{LASER_DWELL_REQUIRED_S:.2f}s"
        return None
    dwell_s = max(0.0, t - float(lock["start_time"]))
    if dwell_s + 1e-9 < LASER_DWELL_REQUIRED_S:
        MATCH_STATE["last_event"] = (
            f"{team} laser dwell {target_name_from_path(target_path)} "
            f"{dwell_s:.2f}/{LASER_DWELL_REQUIRED_S:.2f}s d={distance:.2f}m"
        )
        return None
    dwell_factor = normalized_laser_dwell_factor(dwell_s)
    final_accuracy = max(0.0, min(0.95, accuracy * dwell_factor))
    draw = deterministic_laser_draw(team, target_path, distance, dwell_s)
    reset_laser_lock(team)
    if draw > final_accuracy:
        MATCH_STATE["last_event"] = (
            f"{team} laser miss after dwell: d={distance:.2f}m lateral={lateral:.3f} "
            f"p={final_accuracy:.2f}"
        )
        return None
    MATCH_STATE["last_event"] = (
        f"{team} laser dwell satisfied on {target_name_from_path(target_path)} "
        f"p={final_accuracy:.2f}"
    )
    return target_path


def scripted_fire_after_dwell(team: str, target_path: str) -> bool:
    controller = MATCH_CONTROLLERS.get(team)
    if controller is None:
        return False
    candidate = detect_laser_candidate(team, controller.pose)
    if candidate is None or candidate[0] != target_path:
        MATCH_STATE["last_event"] = f"{team} scripted shot blocked by range/line safety"
        return False
    LASER_LOCKS[team]["target_path"] = target_path
    LASER_LOCKS[team]["start_time"] = float(MATCH_STATE["current_time"]) - LASER_DWELL_FULL_CONFIDENCE_S
    hit_path = update_laser_lock(team, controller.pose, float(MATCH_STATE["current_time"]))
    if hit_path != target_path:
        return False
    result = apply_fire_rule(team, target_path)
    reset_laser_lock(team)
    return result


def remove_next_armor(base_team: str):
    if not BASE_ARMOR[base_team]:
        return
    armor_path = BASE_ARMOR[base_team].pop(0)
    unregister_blocker(armor_path)
    start_pos, start_orient = get_xform(armor_path)
    removed_index = 4 - len(BASE_ARMOR[base_team])
    # Removed armor is lifted out above the wall instead of being dropped onto
    # the floor, so it never becomes a post-hit route obstacle.
    end_pos = (
        start_pos[0],
        start_pos[1],
        BASE_ARMOR_LIFT_CLEARANCE_Z + 0.040 * (removed_index - 1),
    )
    ARMOR_REMOVALS.append(
        {
            "path": armor_path,
            "start_time": float(MATCH_STATE["current_time"]),
            "duration": 0.58,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "orientation": start_orient,
        }
    )
    print(f"[RULE]: {base_team} base armor removed, remaining={len(BASE_ARMOR[base_team])}.")


def knock_down_target(target_path: str):
    target = TARGET_REGISTRY[target_path]
    target["knocked"] = True
    set_visibility(str(target["path"]), False)
    set_visibility(str(target["fall_anim_path"]), True)
    TARGET_FALLS.append(
        {
            "target_path": target_path,
            "start_time": float(MATCH_STATE["current_time"]),
            "duration": 0.65,
        }
    )


def update_target_falls(t: float):
    for fall in list(TARGET_FALLS):
        target = TARGET_REGISTRY[str(fall["target_path"])]
        start_time = float(fall["start_time"])
        duration = float(fall["duration"])
        alpha = max(0.0, min(1.0, (t - start_time) / duration))
        eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
        xy = target["xy"]
        yaw = float(target["yaw"])
        assert isinstance(xy, tuple)
        pitch = -math.radians(86.0) * eased
        set_xform(
            str(target["fall_anim_path"]),
            (xy[0], xy[1], 0.0),
            quat_from_euler(0.0, pitch, yaw),
        )
        if alpha >= 1.0:
            set_visibility(str(target["fall_anim_path"]), False)
            set_visibility(str(target["fallen_path"]), True)
            TARGET_FALLS.remove(fall)


def update_armor_removals(t: float):
    for removal in list(ARMOR_REMOVALS):
        start_time = float(removal["start_time"])
        duration = float(removal["duration"])
        alpha = max(0.0, min(1.0, (t - start_time) / duration))
        eased = 0.5 - 0.5 * math.cos(alpha * math.pi)
        start_pos = removal["start_pos"]
        end_pos = removal["end_pos"]
        orientation = removal["orientation"]
        assert isinstance(start_pos, tuple)
        assert isinstance(end_pos, tuple)
        assert isinstance(orientation, tuple)
        pos = (
            start_pos[0] + (end_pos[0] - start_pos[0]) * eased,
            start_pos[1] + (end_pos[1] - start_pos[1]) * eased,
            start_pos[2] + (end_pos[2] - start_pos[2]) * eased,
        )
        set_xform(str(removal["path"]), pos, orientation)
        if alpha >= 1.0:
            ARMOR_REMOVALS.remove(removal)


def apply_fire_rule(team: str, target_path: str) -> bool:
    target = TARGET_REGISTRY[target_path]
    kind = str(target["kind"])
    owner = str(target["owner"])
    opponent = "blue" if team == "yellow" else "yellow"

    if owner == team:
        MATCH_STATE["last_event"] = f"{team} own-target shot blocked"
        print(f"[RULE]: {team} attempted to shoot own target {target_path.rsplit('/', 1)[-1]}; ignored by safety gate.")
        return False

    if kind == "normal":
        knock_down_target(target_path)
        remove_next_armor(opponent)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 5
        MATCH_STATE["last_event"] = f"{team} hit normal target; {opponent} armor removed"
        print(f"[RULE]: {team} knocked normal target {target_path.rsplit('/', 1)[-1]}.")
        return True

    if kind == f"base_{opponent}":
        knock_down_target(target_path)
        MATCH_STATE[f"score_{team}"] = int(MATCH_STATE[f"score_{team}"]) + 60
        MATCH_STATE["winner"] = team
        MATCH_STATE["last_event"] = f"{team} hit {opponent} base target -> win"
        print(f"[RULE]: {team} knocked {opponent} base target. Match winner={team}.")
        return True
    return False


def apply_target_contact_rule(team: str, target_path: str):
    target = TARGET_REGISTRY[target_path]
    if target["knocked"]:
        return
    target_name = target_path.rsplit("/", 1)[-1]
    MATCH_STATE["last_event"] = f"{team} brushed target {target_name}; target remains standing"
    controller = MATCH_CONTROLLERS.get(team)
    if controller is not None:
        controller.notify_contact(float(MATCH_STATE["current_time"]))
    print(f"[RULE]: {team} contacted target {target_name}; no knockdown, retreat/relocalize required.")


def update_target_contacts(robot_poses: dict[str, tuple[tuple[float, float, float], float]]):
    if MATCH_STATE["winner"] is not None:
        return
    for team, (robot_pos, _yaw) in robot_poses.items():
        robot_xy = (robot_pos[0], robot_pos[1])
        for target_path, target in TARGET_REGISTRY.items():
            if target["knocked"]:
                continue
            xy = target["xy"]
            kind = str(target["kind"])
            assert isinstance(xy, tuple)
            contact_radius = BASE_TARGET_CONTACT_RADIUS if kind.startswith("base_") else TARGET_CONTACT_RADIUS
            if math.hypot(robot_xy[0] - xy[0], robot_xy[1] - xy[1]) <= ROBOT_COLLISION_RADIUS + contact_radius:
                apply_target_contact_rule(team, target_path)
                if MATCH_STATE["winner"] is not None:
                    return


def update_match_rules(t: float, robot_poses: dict[str, tuple[tuple[float, float, float], float]]):
    if MATCH_STATE["winner"] is not None:
        return
    update_target_contacts(robot_poses)
    if MATCH_STATE["winner"] is not None:
        return
    for team, pose in robot_poses.items():
        if t - LAST_FIRE_TIME[team] < FIRE_COOLDOWN:
            reset_laser_lock(team)
            continue
        target_path = update_laser_lock(team, pose, t)
        if target_path is None:
            continue
        LAST_FIRE_TIME[team] = t
        apply_fire_rule(team, target_path)


def export_stage():
    output = Path(args_cli.save_usd) if args_cli.save_usd else Path(__file__).resolve().parent / "output" / "robocup_visionrl_arena.usd"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = get_current_stage()
    stage.GetRootLayer().Export(str(output))
    print(f"[INFO]: Exported USD scene to {output}")
