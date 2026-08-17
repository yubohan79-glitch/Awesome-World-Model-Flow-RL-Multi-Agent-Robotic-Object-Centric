from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from expert_policy import compose_policy_action
from evaluate_policy import actor_action, json_safe, load_policy
from robocup_visionrl_gym_env import (
    BASE_HIT_RADIUS,
    BASE_SHOOT_MIN_RANGE,
    BASE_SHOOT_RANGE,
    BLUE_BASE_XY,
    HALF_ARENA,
    PUSHABLE_OBSTACLE_HALF,
    ROBOT_LENGTH,
    ROBOT_WIDTH,
    YELLOW_BASE_XY,
    active_base_armor_blockers,
    base_attack_pose_quality,
    base_hit_success_cap,
    base_removed_side_lane_quality,
    laser_origin_from_pose,
    wrap_angle,
)
from robocup_visionrl_selfplay_env import (
    AGENTS,
    ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
    TACTICAL_ACTION_DIM,
    RoboCupVisionRLSelfPlayEnv,
    oriented_rect_aabb_collision,
)


MAX_TRANSLATION_PER_STEP_M = 0.12
MAX_YAW_DELTA_PER_STEP_RAD = 0.27
BOUNDARY_TOLERANCE_M = 0.015
STATIC_BLOCKER_TOLERANCE_M = 0.012
PUSHABLE_BOX_TOLERANCE_M = 0.006


@dataclass
class StrictReplayAudit:
    violations: list[dict[str, object]] = field(default_factory=list)
    warnings: list[dict[str, object]] = field(default_factory=list)

    def fail(self, episode: int, step: int, code: str, detail: dict[str, object]):
        payload = {"episode": episode, "step": step, "code": code}
        payload.update(detail)
        self.violations.append(payload)

    def warn(self, episode: int, step: int, code: str, detail: dict[str, object]):
        payload = {"episode": episode, "step": step, "code": code}
        payload.update(detail)
        self.warnings.append(payload)


def target_by_name(env: RoboCupVisionRLSelfPlayEnv, name: str):
    return next((target for target in env.targets if target.name == name), None)


def pose_to_list(pose: np.ndarray) -> list[float]:
    return [round(float(pose[0]), 5), round(float(pose[1]), 5), round(float(pose[2]), 5)]


def static_blocker_penetration(env: RoboCupVisionRLSelfPlayEnv, pose: np.ndarray) -> float:
    x, y = float(pose[0]), float(pose[1])
    max_depth = 0.0
    for center, half_size in env.nav_blockers:
        overlap_x = half_size[0] - abs(x - center[0])
        overlap_y = half_size[1] - abs(y - center[1])
        if overlap_x >= 0.0 and overlap_y >= 0.0:
            max_depth = max(max_depth, min(overlap_x, overlap_y))
    yaw = float(pose[2])
    half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
    half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
    for center, half_size in active_base_armor_blockers(env.armor, inflated=False):
        overlap_x = half_size[0] + half_x - abs(x - center[0])
        overlap_y = half_size[1] + half_y - abs(y - center[1])
        if overlap_x >= 0.0 and overlap_y >= 0.0:
            max_depth = max(max_depth, min(overlap_x, overlap_y))
    return max_depth


def pushable_box_penetration(env: RoboCupVisionRLSelfPlayEnv, pose: np.ndarray) -> tuple[str | None, float]:
    x, y = float(pose[0]), float(pose[1])
    worst_name = None
    worst_depth = 0.0
    for name, center in env.pushable_obstacles.items():
        collided, _normal, penetration = oriented_rect_aabb_collision(
            (x, y),
            float(pose[2]),
            ROBOT_PUSHABLE_VISUAL_HALF_EXTENTS,
            (float(center[0]), float(center[1])),
            (PUSHABLE_OBSTACLE_HALF, PUSHABLE_OBSTACLE_HALF),
        )
        if collided and penetration > worst_depth:
            worst_name = name
            worst_depth = float(penetration)
    return worst_name, worst_depth


def audit_base_win_geometry(
    audit: StrictReplayAudit,
    *,
    episode: int,
    step: int,
    env: RoboCupVisionRLSelfPlayEnv,
    team: str,
    target,
):
    attempt = dict(env.last_shot_attempt.get(team, {}))
    pose = env.poses[team]
    normal_hits = env._normal_hits_against(team)
    base_xy = BLUE_BASE_XY if target.kind == "base_blue" else YELLOW_BASE_XY
    origin = laser_origin_from_pose(pose)
    side_quality = base_removed_side_lane_quality(normal_hits, base_xy, pose[:2])
    pose_quality = base_attack_pose_quality(normal_hits, target.xy, target.yaw, base_xy, pose[:2])
    distance = float(attempt.get("distance_m", math.inf))
    lateral_error = float(attempt.get("lateral_error_m", math.inf))
    accuracy = float(attempt.get("accuracy", math.inf))
    cap = base_hit_success_cap(normal_hits)

    if attempt.get("target") != target.name or not bool(attempt.get("hit", False)):
        audit.fail(episode, step, "base_win_without_matching_shot_attempt", {"team": team, "attempt": attempt})
    if distance < BASE_SHOOT_MIN_RANGE - 1e-6 or distance > BASE_SHOOT_RANGE + 1e-6:
        audit.fail(
            episode,
            step,
            "base_win_distance_out_of_range",
            {"team": team, "target": target.name, "distance_m": round(distance, 5)},
        )
    if lateral_error > BASE_HIT_RADIUS + 1e-6:
        audit.fail(
            episode,
            step,
            "base_win_lateral_error_outside_small_target",
            {"team": team, "target": target.name, "lateral_error_m": round(lateral_error, 5)},
        )
    if side_quality <= 0.0 or pose_quality <= 0.0:
        audit.fail(
            episode,
            step,
            "base_win_from_unopened_or_invalid_side",
            {
                "team": team,
                "target": target.name,
                "normal_hits": normal_hits,
                "pose": pose_to_list(pose),
                "side_quality": round(float(side_quality), 5),
                "pose_quality": round(float(pose_quality), 5),
            },
        )
    if not env._target_line_clear(team, origin, target):
        audit.fail(
            episode,
            step,
            "base_win_laser_blocked_by_armor_or_obstacle",
            {"team": team, "target": target.name, "origin": [round(origin[0], 5), round(origin[1], 5)]},
        )
    if accuracy > cap + 1e-6:
        audit.fail(
            episode,
            step,
            "base_win_probability_cap_exceeded",
            {"team": team, "target": target.name, "accuracy": round(accuracy, 5), "cap": round(cap, 5)},
        )


def audit_step(
    audit: StrictReplayAudit,
    *,
    episode: int,
    step: int,
    env: RoboCupVisionRLSelfPlayEnv,
    previous_poses: dict[str, np.ndarray],
    previous_scores: dict[str, int],
    previous_armor: dict[str, int],
    actions: dict[str, np.ndarray],
    infos: dict[str, dict[str, object]],
):
    for team in AGENTS:
        pose = env.poses[team]
        action = np.asarray(actions[team], dtype=np.float32)
        if action.shape != (TACTICAL_ACTION_DIM,):
            audit.fail(episode, step, "bad_action_shape", {"team": team, "shape": list(action.shape)})
        if not np.isfinite(action).all():
            audit.fail(episode, step, "nonfinite_action", {"team": team})
        if np.max(np.abs(action)) > 1.0001:
            audit.fail(episode, step, "action_out_of_range", {"team": team, "action": action.tolist()})
        if not np.isfinite(pose).all():
            audit.fail(episode, step, "nonfinite_pose", {"team": team, "pose": pose_to_list(pose)})

        prev = previous_poses[team]
        translation = float(np.linalg.norm(pose[:2] - prev[:2]))
        yaw_delta = abs(wrap_angle(float(pose[2] - prev[2])))
        if translation > MAX_TRANSLATION_PER_STEP_M:
            audit.fail(
                episode,
                step,
                "teleport_translation",
                {"team": team, "translation_m": round(translation, 5), "pose": pose_to_list(pose)},
            )
        if yaw_delta > MAX_YAW_DELTA_PER_STEP_RAD:
            audit.fail(
                episode,
                step,
                "teleport_yaw",
                {"team": team, "yaw_delta_rad": round(yaw_delta, 5), "pose": pose_to_list(pose)},
            )

        yaw = float(pose[2])
        half_x = abs(math.cos(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.sin(yaw)) * ROBOT_WIDTH * 0.5
        half_y = abs(math.sin(yaw)) * ROBOT_LENGTH * 0.5 + abs(math.cos(yaw)) * ROBOT_WIDTH * 0.5
        boundary_penetration = max(
            abs(float(pose[0])) + half_x - HALF_ARENA,
            abs(float(pose[1])) + half_y - HALF_ARENA,
        )
        if boundary_penetration > BOUNDARY_TOLERANCE_M:
            audit.fail(
                episode,
                step,
                "arena_boundary_penetration",
                {
                    "team": team,
                    "pose": pose_to_list(pose),
                    "penetration_m": round(float(boundary_penetration), 5),
                },
            )
        if env._static_pose_blocked(pose):
            penetration = static_blocker_penetration(env, pose)
            detail = {"team": team, "pose": pose_to_list(pose), "penetration_m": round(float(penetration), 5)}
            if penetration > STATIC_BLOCKER_TOLERANCE_M:
                audit.fail(episode, step, "static_obstacle_penetration", detail)
            else:
                audit.warn(episode, step, "static_inflation_margin_touch", detail)
        else:
            box_name, penetration = pushable_box_penetration(env, pose)
            if box_name is not None:
                detail = {
                    "team": team,
                    "box": box_name,
                    "pose": pose_to_list(pose),
                    "penetration_m": round(float(penetration), 5),
                }
                if penetration > PUSHABLE_BOX_TOLERANCE_M:
                    audit.fail(episode, step, "pushable_box_penetration", detail)
                else:
                    audit.warn(episode, step, "pushable_box_contact", detail)

        info = infos[team]
        selected = info.get("selected_target")
        if isinstance(selected, str):
            target = target_by_name(env, selected)
            if target is None:
                audit.fail(episode, step, "selected_unknown_target", {"team": team, "target": selected})
            elif target.owner == team:
                audit.fail(episode, step, "selected_own_target", {"team": team, "target": selected})

        for key in ("hit", "own_target_hit", "own_target_blocked", "target_collision"):
            target_name = info.get(key)
            if isinstance(target_name, str):
                target = target_by_name(env, target_name)
                if target is None:
                    audit.fail(episode, step, "event_unknown_target", {"team": team, "event": key, "target": target_name})
                elif key == "hit" and target.owner == team:
                    audit.fail(episode, step, "own_target_fire", {"team": team, "target": target_name})
                elif key in ("own_target_hit", "own_target_blocked"):
                    audit.fail(episode, step, "own_target_fire", {"team": team, "target": target_name})
                elif key == "target_collision":
                    if target.knocked:
                        audit.fail(episode, step, "target_contact_knockdown", {"team": team, "target": target_name})
                    else:
                        audit.warn(episode, step, "target_contact_no_knockdown", {"team": team, "target": target_name})

        if info.get("own_base_hit") or info.get("own_base_blocked") or info.get("own_base_collision"):
            audit.fail(episode, step, "own_base_terminal_error", {"team": team, "info": info})
        if info.get("winner") == team or info.get("simultaneous_base_hit"):
            target_name = str(env.last_shot_attempt.get(team, {}).get("target") or info.get("selected_target", ""))
            target = target_by_name(env, target_name)
            if target is None or not target.kind.startswith("base_"):
                audit.fail(episode, step, "base_win_target_missing", {"team": team, "target": target_name})
            elif target.owner == team:
                audit.fail(episode, step, "base_win_own_target", {"team": team, "target": target.name})
            else:
                audit_base_win_geometry(
                    audit,
                    episode=episode,
                    step=step,
                    env=env,
                    team=team,
                    target=target,
                )

    for team in AGENTS:
        score_delta = env.scores[team] - previous_scores[team]
        armor_delta = previous_armor[team] - env.armor[team]
        if score_delta < 0:
            audit.fail(episode, step, "score_decreased", {"team": team, "delta": score_delta})
        if score_delta % 5 != 0:
            audit.fail(episode, step, "score_not_rule_multiple", {"team": team, "delta": score_delta})
        if score_delta > 65:
            audit.fail(episode, step, "score_jump_too_large", {"team": team, "delta": score_delta})
        if armor_delta < 0:
            audit.fail(episode, step, "armor_increased", {"team": team, "delta": armor_delta})
        if armor_delta > 2:
            audit.fail(episode, step, "armor_drop_too_large", {"team": team, "delta": armor_delta})
        if not 0 <= env.armor[team] <= 4:
            audit.fail(episode, step, "armor_out_of_range", {"team": team, "armor": env.armor[team]})


def run_strict_episode(
    model,
    *,
    episode: int,
    seed: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
    policy_mode: str,
    residual_scale: float,
    trace_writer: csv.DictWriter,
    event_file,
) -> tuple[dict[str, object], StrictReplayAudit]:
    audit = StrictReplayAudit()
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=seed)
    rewards_total = {team: 0.0 for team in AGENTS}
    events = {
        "normal_hits": 0,
        "base_hit_wins": 0,
        "own_target_penalties": 0,
        "blocked_steps": 0,
        "target_contact_events": 0,
        "collision_recovery_events": 0,
        "robot_contacts": 0,
        "block_steps": 0,
        "base_rush_steps": 0,
        "interference_steps": 0,
    }

    steps = 0
    for steps in range(1, max_steps + 1):
        previous_poses = {team: env.poses[team].copy() for team in AGENTS}
        previous_scores = dict(env.scores)
        previous_armor = dict(env.armor)
        actions = {}
        for team in AGENTS:
            raw_action = actor_action(model, observations[team], team, device, deterministic)
            actions[team] = compose_policy_action(
                env,
                team,
                raw_action,
                policy_mode=policy_mode,
                residual_scale=residual_scale,
            )
        observations, rewards, terminations, truncations, infos = env.step(actions)
        audit_step(
            audit,
            episode=episode,
            step=steps,
            env=env,
            previous_poses=previous_poses,
            previous_scores=previous_scores,
            previous_armor=previous_armor,
            actions=actions,
            infos=infos,
        )

        for team in AGENTS:
            rewards_total[team] += float(rewards[team])
            info = infos[team]
            if "hit" in info:
                events["normal_hits"] += 1
            if "winner" in info:
                events["base_hit_wins"] += 1
            if any(
                key in info
                for key in ("own_target_hit", "own_base_hit", "own_target_blocked", "own_base_blocked", "own_base_collision")
            ):
                events["own_target_penalties"] += 1
            if info.get("blocked"):
                events["blocked_steps"] += 1
            if info.get("pushed_obstacle"):
                events["push_events"] = events.get("push_events", 0) + 1
            if info.get("target_collision"):
                events["target_contact_events"] += 1
            if info.get("relocalizing"):
                events["collision_recovery_events"] += 1
            if info.get("robot_contact"):
                events["robot_contacts"] += 1
            if info.get("tactic") == "block":
                events["block_steps"] += 1
            if info.get("base_rush"):
                events["base_rush_steps"] += 1
            if info.get("interference"):
                events["interference_steps"] += 1

            trace_writer.writerow(
                {
                    "episode": episode,
                    "seed": seed,
                    "step": steps,
                    "elapsed_s": round(float(env.elapsed), 3),
                    "team": team,
                    "x": round(float(env.poses[team][0]), 5),
                    "y": round(float(env.poses[team][1]), 5),
                    "yaw": round(float(env.poses[team][2]), 5),
                    "a_target": round(float(actions[team][0]), 5),
                    "a_base_rush": round(float(actions[team][1]), 5),
                    "a_block": round(float(actions[team][2]), 5),
                    "a_recover": round(float(actions[team][3]), 5),
                    "a_fire": round(float(actions[team][4]), 5),
                    "a_risk": round(float(actions[team][5]), 5),
                    "tactic": info.get("tactic", ""),
                    "selected_target": info.get("selected_target", ""),
                    "fire_ready": bool(info.get("fire_ready", False)),
                    "blocked": bool(info.get("blocked", False)),
                    "holding_fire_pose": bool(info.get("holding_fire_pose", False)),
                    "pushed_obstacle": info.get("pushed_obstacle", ""),
                    "score_yellow": env.scores["yellow"],
                    "score_blue": env.scores["blue"],
                    "armor_yellow": env.armor["yellow"],
                    "armor_blue": env.armor["blue"],
                    "localization_confidence": round(float(env.localization_confidence[team]), 5),
                    "box_ne_x": round(float(env.pushable_obstacles["box_ne"][0]), 5),
                    "box_ne_y": round(float(env.pushable_obstacles["box_ne"][1]), 5),
                    "box_sw_x": round(float(env.pushable_obstacles["box_sw"][0]), 5),
                    "box_sw_y": round(float(env.pushable_obstacles["box_sw"][1]), 5),
                }
            )

        if any(
            key in infos[team]
            for team in AGENTS
            for key in (
                "hit",
                "winner",
                "own_target_hit",
                "own_target_blocked",
                "own_base_hit",
                "own_base_blocked",
                "own_base_collision",
                "target_collision",
                "robot_contact",
            )
        ):
            event_file.write(
                json.dumps(
                    json_safe(
                    {
                        "episode": episode,
                        "step": steps,
                        "elapsed_s": round(float(env.elapsed), 3),
                        "scores": dict(env.scores),
                        "armor": dict(env.armor),
                        "yellow_info": {k: v for k, v in infos["yellow"].items() if k != "action_labels"},
                        "blue_info": {k: v for k, v in infos["blue"].items() if k != "action_labels"},
                    }
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

        if any(terminations.values()) or any(truncations.values()):
            break

    episode_summary = {
        "episode": episode,
        "seed": seed,
        "winner": env.winner or "timeout",
        "elapsed_s": round(float(env.elapsed), 3),
        "steps": steps,
        "scores": dict(env.scores),
        "armor": dict(env.armor),
        "rewards": {team: round(float(value), 3) for team, value in rewards_total.items()},
        "events": events,
        "target_order": {team: list(env.target_order[team]) for team in AGENTS},
        "violations": len(audit.violations),
        "warnings": len(audit.warnings),
    }
    return episode_summary, audit


def summarize(episodes: list[dict[str, object]], audits: list[StrictReplayAudit], wall_time_s: float) -> dict[str, object]:
    count = len(episodes)
    winners = [str(item["winner"]) for item in episodes]
    totals: dict[str, int] = {}
    for episode in episodes:
        for key, value in episode["events"].items():
            totals[key] = totals.get(key, 0) + int(value)
    return {
        "episodes": count,
        "yellow_win_rate": round(winners.count("yellow") / count, 4),
        "blue_win_rate": round(winners.count("blue") / count, 4),
        "draw_or_timeout_rate": round((winners.count("draw") + winners.count("timeout")) / count, 4),
        "hard_violations": sum(len(audit.violations) for audit in audits),
        "warnings": sum(len(audit.warnings) for audit in audits),
        "normal_hits_per_episode": round(totals.get("normal_hits", 0) / count, 4),
        "base_wins_per_episode": round(totals.get("base_hit_wins", 0) / count, 4),
        "own_target_penalties_per_episode": round(totals.get("own_target_penalties", 0) / count, 4),
        "blocked_steps_per_episode": round(totals.get("blocked_steps", 0) / count, 4),
        "target_contact_events_per_episode": round(totals.get("target_contact_events", 0) / count, 4),
        "robot_contacts_per_episode": round(totals.get("robot_contacts", 0) / count, 4),
        "recovery_events_per_episode": round(totals.get("collision_recovery_events", 0) / count, 4),
        "block_steps_per_episode": round(totals.get("block_steps", 0) / count, 4),
        "base_rush_steps_per_episode": round(totals.get("base_rush_steps", 0) / count, 4),
        "wall_time_s": round(wall_time_s, 3),
    }


def write_markdown_report(payload: dict[str, object], report_path: Path):
    summary = payload["summary"]
    verdict = "PASS" if int(summary["hard_violations"]) == 0 else "FAIL"
    lines = [
        "# Strict SAC Flow Replay Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "This report replays the trained object-centric SAC Flow tactical actor and audits each step against strict rule and physics invariants.",
        "",
        "## Replay Setup",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- deterministic: `{payload['deterministic']}`",
        f"- device: `{payload['device']}`",
        f"- episodes: `{summary['episodes']}`",
        f"- max step translation: `{MAX_TRANSLATION_PER_STEP_M} m`",
        f"- max step yaw delta: `{MAX_YAW_DELTA_PER_STEP_RAD} rad`",
        f"- static blocker tolerance: `{STATIC_BLOCKER_TOLERANCE_M} m`",
        "",
        "## Strict Checks",
        "",
        "- action shape is exactly 6D and bounded to [-1, 1]",
        "- robot pose is finite, inside the arena boundary, and outside static blockers",
        "- per-step translation/yaw changes stay within differential-drive limits",
        "- selected targets and fired targets must belong to the opponent",
        "- own-base hit/collision is an immediate hard violation",
        "- scores and armor only change in rule-compatible directions",
        "- target contact is recorded as a warning unless it actually knocks down a target",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON summary: `{payload['output_files']['summary_json']}`",
            f"- CSV trace: `{payload['output_files']['trace_csv']}`",
            f"- JSONL event log: `{payload['output_files']['events_jsonl']}`",
            "",
            "## Notes",
            "",
            "Blocked steps are not counted as hard violations because the costmap/barrier logic prevented penetration. Actual penetration after integration is a hard violation.",
            "A pose that only touches the inflated costmap boundary within the static-blocker tolerance is counted as a warning, not as physical wall penetration.",
            "Pushable-box contact is allowed only within the tolerance; robot-box penetration is a hard violation.",
            "Target contact is allowed only as a non-scoring brush/contact event; any contact-induced knockdown remains a hard violation.",
            "Robot-robot contact is allowed as a tactical event, but it is counted so future training can penalize unsafe or wasteful contact.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repo_relative(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description="Strictly replay and audit a trained SAC Flow policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--policy-mode", choices=("auto", "direct", "expert", "residual_expert"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("../output/replay/world_model_sacflow_strict_replay_abs"))
    parser.add_argument("--report", type=Path, default=Path("../../docs/rl_strict_replay_audit.md"))
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir.is_absolute() else (script_dir / args.output_dir).resolve()
    report_path = args.report if args.report.is_absolute() else (script_dir / args.report).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, checkpoint = load_policy(args.checkpoint, device)
    train_config = checkpoint.get("config", {})
    policy_mode = str(train_config.get("policy_mode", "direct")) if args.policy_mode == "auto" else args.policy_mode
    residual_scale = (
        float(train_config.get("residual_scale", 0.28))
        if args.residual_scale is None
        else float(args.residual_scale)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "strict_replay_summary.json"
    trace_csv = output_dir / "strict_replay_trace.csv"
    events_jsonl = output_dir / "strict_replay_events.jsonl"

    started = time.perf_counter()
    episodes = []
    audits = []
    with trace_csv.open("w", newline="", encoding="utf-8") as trace_handle, events_jsonl.open("w", encoding="utf-8") as event_handle:
        fieldnames = [
            "episode",
            "seed",
            "step",
            "elapsed_s",
            "team",
            "x",
            "y",
            "yaw",
            "a_target",
            "a_base_rush",
            "a_block",
            "a_recover",
            "a_fire",
            "a_risk",
            "tactic",
            "selected_target",
            "fire_ready",
            "blocked",
            "holding_fire_pose",
            "pushed_obstacle",
            "score_yellow",
            "score_blue",
            "armor_yellow",
            "armor_blue",
            "localization_confidence",
            "box_ne_x",
            "box_ne_y",
            "box_sw_x",
            "box_sw_y",
        ]
        writer = csv.DictWriter(trace_handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode_index in range(args.episodes):
            episode_summary, audit = run_strict_episode(
                model,
                episode=episode_index,
                seed=args.seed + episode_index,
                max_steps=args.max_steps,
                device=device,
                deterministic=not args.stochastic,
                policy_mode=policy_mode,
                residual_scale=residual_scale,
                trace_writer=writer,
                event_file=event_handle,
            )
            episodes.append(episode_summary)
            audits.append(audit)

    wall_time_s = time.perf_counter() - started
    payload = {
        "summary": summarize(episodes, audits, wall_time_s),
        "checkpoint": str(args.checkpoint),
        "deterministic": not args.stochastic,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_config": checkpoint.get("config", {}),
        "policy_mode": policy_mode,
        "residual_scale": residual_scale,
        "episodes": episodes,
        "violations": [item for audit in audits for item in audit.violations],
        "warnings": [item for audit in audits for item in audit.warnings],
        "output_files": {
            "summary_json": repo_relative(summary_json),
            "trace_csv": repo_relative(trace_csv),
            "events_jsonl": repo_relative(events_jsonl),
        },
    }
    payload = json_safe(payload)
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(payload, report_path)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[INFO] wrote {summary_json}")
    print(f"[INFO] wrote {report_path}")
    if payload["summary"]["hard_violations"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
