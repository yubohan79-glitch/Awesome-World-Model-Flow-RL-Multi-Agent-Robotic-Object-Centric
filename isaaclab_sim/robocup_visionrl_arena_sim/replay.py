from __future__ import annotations

import csv
import json
import math
from pathlib import Path


from ._bootstrap import (
    ARENA_SIZE,
    BASE_ARMOR,
    MATCH_STATE,
    NAV_BLOCKERS,
    PUSHABLE_OBSTACLES,
    PUSHABLE_OBSTACLE_HALF,
    PUSH_OBSTACLE_CLEARANCE,
    TARGET_REGISTRY,
    TRAINED_REPLAY,
    args_cli
)
from .costmap import (
    clamp_to_arena,
    dynamic_pushable_costmap,
    dynamic_target_costmap,
    robot_pushable_collision,
    warn_costmap,
    wrap_angle
)
from .laser import apply_fire_rule
from .spawn import segment_intersects_aabb, target_path_from_name
from .transforms import get_xform, quat_from_euler, set_xform

def load_trained_replay():
    if TRAINED_REPLAY["loaded"]:
        return
    if not args_cli.replay_trace:
        TRAINED_REPLAY["loaded"] = True
        return

    trace_path = Path(args_cli.replay_trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"Replay trace not found: {trace_path}")

    rows: dict[str, list[dict[str, object]]] = {"yellow": [], "blue": []}
    last_time = 0.0
    with trace_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["episode"]) != int(args_cli.replay_episode):
                continue
            team = str(row["team"])
            if team not in rows:
                continue
            item = {
                "t": float(row["elapsed_s"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "yaw": float(row["yaw"]),
                "selected_target": str(row.get("selected_target", "")),
                "tactic": str(row.get("tactic", "")),
                "fire_ready": str(row.get("fire_ready", "")).lower() == "true",
                "score_yellow": int(float(row.get("score_yellow", 0) or 0)),
                "score_blue": int(float(row.get("score_blue", 0) or 0)),
                "armor_yellow": int(float(row.get("armor_yellow", 4) or 4)),
                "armor_blue": int(float(row.get("armor_blue", 4) or 4)),
                "localization_confidence": float(row.get("localization_confidence", 1.0) or 1.0),
                "box_ne": (
                    float(row["box_ne_x"]),
                    float(row["box_ne_y"]),
                )
                if row.get("box_ne_x") not in (None, "")
                else None,
                "box_sw": (
                    float(row["box_sw_x"]),
                    float(row["box_sw_y"]),
                )
                if row.get("box_sw_x") not in (None, "")
                else None,
            }
            rows[team].append(item)
            last_time = max(last_time, float(item["t"]))

    for team_rows in rows.values():
        team_rows.sort(key=lambda item: float(item["t"]))
    if not rows["yellow"] or not rows["blue"]:
        raise RuntimeError(f"No replay rows found for episode {args_cli.replay_episode} in {trace_path}")

    events: list[dict[str, object]] = []
    events_path = Path(args_cli.replay_events) if args_cli.replay_events else None
    if events_path is not None and events_path.exists():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if int(payload.get("episode", -1)) != int(args_cli.replay_episode):
                    continue
                event_time = float(payload.get("elapsed_s", 0.0))
                for team in ("yellow", "blue"):
                    info = payload.get(f"{team}_info", {})
                    if not isinstance(info, dict):
                        continue
                    hit_target = info.get("hit")
                    if isinstance(hit_target, str) and hit_target:
                        events.append({"t": event_time, "team": team, "target": hit_target, "kind": "hit"})
                    winner = info.get("winner")
                    selected = info.get("selected_target")
                    if winner == team and isinstance(selected, str) and selected.endswith("BaseTarget"):
                        events.append({"t": event_time, "team": team, "target": selected, "kind": "base_win"})

    seen: set[tuple[float, str, str, str]] = set()
    unique_events = []
    for event in sorted(events, key=lambda item: (float(item["t"]), str(item["team"]), str(item["target"]))):
        key = (round(float(event["t"]), 3), str(event["team"]), str(event["target"]), str(event["kind"]))
        if key in seen:
            continue
        seen.add(key)
        unique_events.append(event)

    TRAINED_REPLAY["rows"] = rows
    TRAINED_REPLAY["events"] = unique_events
    TRAINED_REPLAY["applied_events"] = set()
    TRAINED_REPLAY["last_time"] = last_time
    TRAINED_REPLAY["render_poses"] = {"yellow": None, "blue": None}
    TRAINED_REPLAY["last_box_trace_xy"] = {}
    reset_pushable_obstacles()
    TRAINED_REPLAY["loaded"] = True
    print(
        f"[REPLAY]: loaded trained policy trace episode={args_cli.replay_episode} "
        f"duration={last_time:.1f}s events={len(unique_events)} from {trace_path}"
    )


def replay_row_at(team: str, t: float) -> dict[str, object]:
    load_trained_replay()
    rows = TRAINED_REPLAY["rows"]
    assert isinstance(rows, dict)
    team_rows = rows[team]
    assert isinstance(team_rows, list)
    if t <= float(team_rows[0]["t"]):
        return team_rows[0]
    for index in range(len(team_rows) - 1):
        row0 = team_rows[index]
        row1 = team_rows[index + 1]
        t0 = float(row0["t"])
        t1 = float(row1["t"])
        if t <= t1:
            alpha = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            yaw0 = float(row0["yaw"])
            yaw1 = yaw0 + wrap_angle(float(row1["yaw"]) - yaw0)

            def interp_box(name: str):
                box0 = row0.get(name)
                box1 = row1.get(name)
                if not isinstance(box0, tuple) or not isinstance(box1, tuple):
                    return box1
                return (
                    float(box0[0]) + (float(box1[0]) - float(box0[0])) * alpha,
                    float(box0[1]) + (float(box1[1]) - float(box0[1])) * alpha,
                )

            return {
                "t": t,
                "x": float(row0["x"]) + (float(row1["x"]) - float(row0["x"])) * alpha,
                "y": float(row0["y"]) + (float(row1["y"]) - float(row0["y"])) * alpha,
                "yaw": wrap_angle(yaw0 + (yaw1 - yaw0) * alpha),
                "selected_target": row1.get("selected_target", ""),
                "tactic": row1.get("tactic", ""),
                "fire_ready": bool(row1.get("fire_ready", False)),
                "score_yellow": row1.get("score_yellow", MATCH_STATE["score_yellow"]),
                "score_blue": row1.get("score_blue", MATCH_STATE["score_blue"]),
                "armor_yellow": row1.get("armor_yellow", len(BASE_ARMOR["yellow"])),
                "armor_blue": row1.get("armor_blue", len(BASE_ARMOR["blue"])),
                "localization_confidence": row1.get("localization_confidence", 1.0),
                "box_ne": interp_box("box_ne"),
                "box_sw": interp_box("box_sw"),
            }
    return team_rows[-1]


def apply_trained_replay_events(t: float):
    load_trained_replay()
    applied = TRAINED_REPLAY["applied_events"]
    events = TRAINED_REPLAY["events"]
    assert isinstance(applied, set)
    assert isinstance(events, list)
    for index, event in enumerate(events):
        if index in applied or float(event["t"]) > t:
            continue
        target_path = target_path_from_name(str(event["target"]))
        target = TARGET_REGISTRY.get(target_path)
        if target is None:
            raise RuntimeError(f"Replay event target not found: {event['target']}")
        if not bool(target["knocked"]):
            apply_fire_rule(str(event["team"]), target_path)
        applied.add(index)


def trained_replay_pushable_pose(
    team: str,
    proposed_pos: tuple[float, float, float],
    yaw: float,
) -> tuple[float, float, float]:
    render_poses = TRAINED_REPLAY["render_poses"]
    assert isinstance(render_poses, dict)
    previous = render_poses.get(team)
    if isinstance(previous, tuple):
        prev_xy = (float(previous[0]), float(previous[1]))
    else:
        prev_xy = (proposed_pos[0], proposed_pos[1])

    motion = (proposed_pos[0] - prev_xy[0], proposed_pos[1] - prev_xy[1])
    motion_norm = math.hypot(motion[0], motion[1])
    if motion_norm > 1e-5:
        motion_dir = (motion[0] / motion_norm, motion[1] / motion_norm)
    else:
        motion_dir = (math.cos(yaw), math.sin(yaw))

    corrected = (proposed_pos[0], proposed_pos[1])
    for _pass in range(8):
        changed = False
        for path, obstacle in PUSHABLE_OBSTACLES.items():
            xy = obstacle["xy"]
            half = obstacle["half"]
            z = float(obstacle["z"])
            assert isinstance(xy, list)
            assert isinstance(half, tuple)
            center = (float(xy[0]), float(xy[1]))
            collided, normal, penetration = robot_pushable_collision(corrected, yaw, center, half)
            if not collided:
                continue

            away_from_robot = (-normal[0], -normal[1])
            push_dir = motion_dir if motion_dir[0] * away_from_robot[0] + motion_dir[1] * away_from_robot[1] > 0.20 else away_from_robot
            push_norm = math.hypot(push_dir[0], push_dir[1])
            if push_norm <= 1e-8:
                push_dir = away_from_robot
                push_norm = max(1e-8, math.hypot(push_dir[0], push_dir[1]))
            push_dir = (push_dir[0] / push_norm, push_dir[1] / push_norm)
            push_step = min(0.090, max(0.020, penetration + 0.012))
            candidate = (center[0] + push_dir[0] * push_step, center[1] + push_dir[1] * push_step)

            if pushable_position_valid(path, candidate):
                xy[0], xy[1] = candidate
                set_xform(path, (candidate[0], candidate[1], z), quat_from_euler(0.0, 0.0, 0.0))
                warn_costmap("trained_replay", f"robot pushed {path.rsplit('/', 1)[-1]} to ({candidate[0]:.2f}, {candidate[1]:.2f})")
                changed = True
                continue

            # Jammed box: keep the rendered robot on the near side of the box
            # instead of letting the trace visually pass through it.
            corrected = (
                corrected[0] + normal[0] * (penetration + 0.010),
                corrected[1] + normal[1] * (penetration + 0.010),
            )
            corrected = clamp_to_arena(corrected)
            warn_costmap("trained_replay", f"{path.rsplit('/', 1)[-1]} jammed; robot held outside box")
            changed = True
        if not changed:
            break
    for _pass in range(4):
        separated = True
        for path, obstacle in PUSHABLE_OBSTACLES.items():
            xy = obstacle["xy"]
            half = obstacle["half"]
            assert isinstance(xy, list)
            assert isinstance(half, tuple)
            collided, normal, penetration = robot_pushable_collision(corrected, yaw, (float(xy[0]), float(xy[1])), half)
            if not collided:
                continue
            corrected = (
                corrected[0] + normal[0] * (penetration + 0.014),
                corrected[1] + normal[1] * (penetration + 0.014),
            )
            corrected = clamp_to_arena(corrected)
            warn_costmap("trained_replay", f"{path.rsplit('/', 1)[-1]} visual hull separated after trace correction")
            separated = False
        if separated:
            break
    render_poses[team] = (corrected[0], corrected[1], proposed_pos[2])
    return (corrected[0], corrected[1], proposed_pos[2])


def apply_replay_box_positions(row: dict[str, object]) -> bool:
    aliases = {
        "box_ne": "RandomObstacleNorthEast",
        "box_sw": "RandomObstacleSouthWest",
    }
    applied = False
    last_trace_xy = TRAINED_REPLAY.setdefault("last_box_trace_xy", {})
    assert isinstance(last_trace_xy, dict)
    for column_name, prim_suffix in aliases.items():
        xy = row.get(column_name)
        if not isinstance(xy, tuple):
            continue
        previous_xy = last_trace_xy.get(column_name)
        trace_changed = (
            not isinstance(previous_xy, tuple)
            or math.hypot(float(xy[0]) - float(previous_xy[0]), float(xy[1]) - float(previous_xy[1])) > 1e-5
        )
        last_trace_xy[column_name] = (float(xy[0]), float(xy[1]))
        if not trace_changed:
            continue
        target_path = ""
        for path in PUSHABLE_OBSTACLES:
            if path.rsplit("/", 1)[-1] == prim_suffix:
                target_path = path
                break
        if not target_path:
            continue
        obstacle = PUSHABLE_OBSTACLES[target_path]
        stored_xy = obstacle["xy"]
        z = float(obstacle["z"])
        assert isinstance(stored_xy, list)
        stored_xy[0], stored_xy[1] = float(xy[0]), float(xy[1])
        set_xform(target_path, (stored_xy[0], stored_xy[1], z), quat_from_euler(0.0, 0.0, 0.0))
        applied = True
    return applied


def point_blocked(point: tuple[float, float]) -> bool:
    x, y = point
    for _, center, half_size in NAV_BLOCKERS:
        if abs(x - center[0]) <= half_size[0] and abs(y - center[1]) <= half_size[1]:
            return True
    return False


def segment_blocked(p0: tuple[float, float], p1: tuple[float, float]) -> bool:
    for _, center, half_size in NAV_BLOCKERS:
        if segment_intersects_aabb(p0, p1, center, half_size):
            return True
    return False


def pushable_collision_path(point: tuple[float, float]) -> str | None:
    for path, center, half_size in dynamic_pushable_costmap():
        if abs(point[0] - center[0]) <= half_size[0] and abs(point[1] - center[1]) <= half_size[1]:
            return path
    return None


def pushable_position_valid(path: str, xy: tuple[float, float]) -> bool:
    limit = ARENA_SIZE * 0.5 - PUSHABLE_OBSTACLE_HALF - PUSH_OBSTACLE_CLEARANCE
    if not (-limit <= xy[0] <= limit and -limit <= xy[1] <= limit):
        return False
    inflated = PUSHABLE_OBSTACLE_HALF + PUSH_OBSTACLE_CLEARANCE
    for _blocker_path, center, half_size in NAV_BLOCKERS:
        if abs(xy[0] - center[0]) <= half_size[0] + inflated and abs(xy[1] - center[1]) <= half_size[1] + inflated:
            return False
    for target_path, center, radius in dynamic_target_costmap():
        if math.hypot(xy[0] - center[0], xy[1] - center[1]) < radius + inflated:
            return False
    for other_path, center, half_size in dynamic_pushable_costmap():
        if other_path == path:
            continue
        if abs(xy[0] - center[0]) <= half_size[0] + inflated and abs(xy[1] - center[1]) <= half_size[1] + inflated:
            return False
    return True


def reset_pushable_obstacles():
    for path, obstacle in PUSHABLE_OBSTACLES.items():
        start_xy = obstacle.get("start_xy")
        if not isinstance(start_xy, tuple):
            continue
        xy = obstacle["xy"]
        z = float(obstacle["z"])
        assert isinstance(xy, list)
        xy[0], xy[1] = float(start_xy[0]), float(start_xy[1])
        set_xform(path, (xy[0], xy[1], z), quat_from_euler(0.0, 0.0, 0.0))


def sync_pushable_obstacles_from_stage():
    for path, obstacle in PUSHABLE_OBSTACLES.items():
        xy = obstacle["xy"]
        assert isinstance(xy, list)
        try:
            translation, _orientation = get_xform(path)
        except RuntimeError:
            continue
        if not (math.isfinite(translation[0]) and math.isfinite(translation[1])):
            continue
        if abs(translation[0]) > ARENA_SIZE or abs(translation[1]) > ARENA_SIZE:
            continue
        xy[0], xy[1] = float(translation[0]), float(translation[1])
