from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from expert_policy import compose_policy_action
from evaluate_policy import actor_action, json_safe, load_policy
from replay_policy_strict import pushable_box_penetration, static_blocker_penetration
from robocup_visionrl_gym_env import (
    BASE_HIT_SUCCESS_BY_NORMAL_HITS,
    PUSHABLE_OBSTACLE_STARTS,
    wrap_angle,
)
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


ROOT = Path(__file__).resolve().parents[2]
ABNORMAL_SPIN_YAW_DELTA_RAD = 0.11
ABNORMAL_SPIN_TRANSLATION_M = 0.006


def normal_hits_before_base(order: list[str]) -> int | None:
    hits = 0
    for name in order:
        if name.endswith("BaseTarget"):
            return hits
        hits += 1
    return None


def csv_safe_counter(counter: Counter[int]) -> str:
    return ";".join(f"{key}:{counter.get(key, 0)}" for key in range(1, 5))


def run_episode(
    model,
    *,
    seed: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
    policy_mode: str,
    residual_scale: float,
) -> dict[str, object]:
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=seed)
    rewards_total = {team: 0.0 for team in AGENTS}
    metrics = {
        "normal_hits": {team: 0 for team in AGENTS},
        "base_wins": {team: 0 for team in AGENTS},
        "base_attempts_by_hits": {team: Counter() for team in AGENTS},
        "base_wins_by_hits": {team: Counter() for team in AGENTS},
        "push_events": {team: 0 for team in AGENTS},
        "robot_contacts": 0,
        "relocalization_events": {team: 0 for team in AGENTS},
        "abnormal_spin_steps": {team: 0 for team in AGENTS},
        "target_contact_events": {team: 0 for team in AGENTS},
        "static_penetrations": {team: 0 for team in AGENTS},
        "box_penetrations": {team: 0 for team in AGENTS},
        "repeat_target_order_events": {team: 0 for team in AGENTS},
    }
    previous_boxes = {name: value.copy() for name, value in env.pushable_obstacles.items()}
    max_box_displacement = {name: 0.0 for name in env.pushable_obstacles}
    final_box_displacement = {name: 0.0 for name in env.pushable_obstacles}
    base_attempt_seen: set[tuple[str, str, int, float, str]] = set()
    base_rush_episode_seen: set[tuple[str, int]] = set()

    steps = 0
    for steps in range(1, max_steps + 1):
        previous_poses = {team: env.poses[team].copy() for team in AGENTS}
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
        for team in AGENTS:
            rewards_total[team] += float(rewards[team])
            info = infos[team]
            if info.get("hit"):
                metrics["normal_hits"][team] += 1
            if info.get("winner") == team:
                metrics["base_wins"][team] += 1
                hits_before = max(0, min(4, env.strategy_counts[team]["normal_hits"]))
                metrics["base_wins_by_hits"][team][hits_before] += 1
            if info.get("pushed_obstacle"):
                metrics["push_events"][team] += 1
            if info.get("robot_contact"):
                metrics["robot_contacts"] += 1
            if info.get("relocalizing"):
                metrics["relocalization_events"][team] += 1
            if info.get("target_collision"):
                metrics["target_contact_events"][team] += 1
            if info.get("base_rush") or str(info.get("selected_target", "")).endswith("BaseTarget"):
                hits_before = max(0, min(4, env.strategy_counts[team]["normal_hits"]))
                episode_key = (team, hits_before)
                if episode_key not in base_rush_episode_seen:
                    base_rush_episode_seen.add(episode_key)
                    metrics["base_attempts_by_hits"][team][hits_before] += 1

            pose = env.poses[team]
            translation = float(np.linalg.norm(pose[:2] - previous_poses[team][:2]))
            yaw_delta = abs(wrap_angle(float(pose[2] - previous_poses[team][2])))
            if (
                yaw_delta > ABNORMAL_SPIN_YAW_DELTA_RAD
                and translation < ABNORMAL_SPIN_TRANSLATION_M
                and info.get("tactic") not in ("recover", "attack")
            ):
                metrics["abnormal_spin_steps"][team] += 1
            if static_blocker_penetration(env, pose) > 0.012:
                metrics["static_penetrations"][team] += 1
            _box_name, box_depth = pushable_box_penetration(env, pose)
            if box_depth > 0.006:
                metrics["box_penetrations"][team] += 1

            shot_attempt = info.get("shot_attempt", {})
            if isinstance(shot_attempt, dict) and str(shot_attempt.get("target", "")).endswith("BaseTarget"):
                reason = str(shot_attempt.get("reason", ""))
                if reason != "dwell":
                    hits_before = max(0, min(4, env.strategy_counts[team]["normal_hits"]))
                    key = (
                        team,
                        str(shot_attempt.get("target", "")),
                        hits_before,
                        float(shot_attempt.get("dwell_s", 0.0) or 0.0),
                        reason,
                    )
                    episode_key = (team, hits_before)
                    if episode_key not in base_rush_episode_seen and key not in base_attempt_seen:
                        base_attempt_seen.add(key)
                        metrics["base_attempts_by_hits"][team][hits_before] += 1

        for name, xy in env.pushable_obstacles.items():
            step_displacement = float(np.linalg.norm(xy - previous_boxes[name]))
            if step_displacement > 1e-5:
                previous_boxes[name] = xy.copy()
            start = PUSHABLE_OBSTACLE_STARTS[name]
            displacement = float(np.linalg.norm(xy - start))
            max_box_displacement[name] = max(max_box_displacement[name], displacement)
            final_box_displacement[name] = displacement

        if any(terminations.values()) or any(truncations.values()):
            break

    for team in AGENTS:
        order = list(env.target_order[team])
        metrics["repeat_target_order_events"][team] = len(order) - len(set(order))

    return {
        "seed": seed,
        "winner": env.winner or "timeout",
        "elapsed_s": round(float(env.elapsed), 3),
        "steps": steps,
        "scores": dict(env.scores),
        "armor": dict(env.armor),
        "rewards": {team: round(float(value), 3) for team, value in rewards_total.items()},
        "target_order": {team: list(env.target_order[team]) for team in AGENTS},
        "normal_hits": {team: int(metrics["normal_hits"][team]) for team in AGENTS},
        "normal_hits_before_base": {team: normal_hits_before_base(list(env.target_order[team])) for team in AGENTS},
        "base_wins": {team: int(metrics["base_wins"][team]) for team in AGENTS},
        "base_attempts_by_hits": {
            team: {str(key): int(value) for key, value in metrics["base_attempts_by_hits"][team].items()}
            for team in AGENTS
        },
        "base_wins_by_hits": {
            team: {str(key): int(value) for key, value in metrics["base_wins_by_hits"][team].items()}
            for team in AGENTS
        },
        "push_events": {team: int(metrics["push_events"][team]) for team in AGENTS},
        "robot_contacts": int(metrics["robot_contacts"]),
        "relocalization_events": {team: int(metrics["relocalization_events"][team]) for team in AGENTS},
        "abnormal_spin_steps": {team: int(metrics["abnormal_spin_steps"][team]) for team in AGENTS},
        "target_contact_events": {team: int(metrics["target_contact_events"][team]) for team in AGENTS},
        "static_penetrations": {team: int(metrics["static_penetrations"][team]) for team in AGENTS},
        "box_penetrations": {team: int(metrics["box_penetrations"][team]) for team in AGENTS},
        "repeat_target_order_events": {team: int(metrics["repeat_target_order_events"][team]) for team in AGENTS},
        "final_box_displacement_m": {name: round(float(value), 4) for name, value in final_box_displacement.items()},
        "max_box_displacement_m": {name: round(float(value), 4) for name, value in max_box_displacement.items()},
    }


def summarize(episodes: list[dict[str, object]], wall_time_s: float) -> dict[str, object]:
    count = len(episodes)
    winners = [str(episode["winner"]) for episode in episodes]
    hit_distribution = {team: Counter() for team in AGENTS}
    base_attempts_by_hits = {team: Counter() for team in AGENTS}
    base_wins_by_hits = {team: Counter() for team in AGENTS}
    for episode in episodes:
        for team in AGENTS:
            hit_count = int(episode["normal_hits"][team])
            hit_distribution[team][max(1, min(4, hit_count))] += 1
            for key, value in episode["base_attempts_by_hits"][team].items():
                base_attempts_by_hits[team][int(key)] += int(value)
            for key, value in episode["base_wins_by_hits"][team].items():
                base_wins_by_hits[team][int(key)] += int(value)

    base_success_by_hits = {}
    for team in AGENTS:
        base_success_by_hits[team] = {}
        for hits in range(1, 5):
            wins = base_wins_by_hits[team][hits]
            attempts = max(base_attempts_by_hits[team][hits], wins)
            base_success_by_hits[team][str(hits)] = {
                "attempts": attempts,
                "wins": wins,
                "success_rate": round(wins / attempts, 4) if attempts else 0.0,
                "configured_cap": BASE_HIT_SUCCESS_BY_NORMAL_HITS[hits],
            }

    def mean_team_metric(name: str, team: str) -> float:
        return float(np.mean([float(episode[name][team]) for episode in episodes]))

    return {
        "episodes": count,
        "yellow_win_rate": round(winners.count("yellow") / count, 4),
        "blue_win_rate": round(winners.count("blue") / count, 4),
        "draw_rate": round((winners.count("draw") + winners.count("timeout")) / count, 4),
        "mean_episode_time_s": round(float(np.mean([float(ep["elapsed_s"]) for ep in episodes])), 4),
        "mean_yellow_score": round(float(np.mean([int(ep["scores"]["yellow"]) for ep in episodes])), 4),
        "mean_blue_score": round(float(np.mean([int(ep["scores"]["blue"]) for ep in episodes])), 4),
        "mean_normal_hits_yellow": round(mean_team_metric("normal_hits", "yellow"), 4),
        "mean_normal_hits_blue": round(mean_team_metric("normal_hits", "blue"), 4),
        "normal_hit_count_distribution": {
            team: {str(key): round(hit_distribution[team][key] / count, 4) for key in range(1, 5)}
            for team in AGENTS
        },
        "base_success_by_hits": base_success_by_hits,
        "push_events_per_episode": {
            team: round(mean_team_metric("push_events", team), 4)
            for team in AGENTS
        },
        "mean_final_box_displacement_m": {
            name: round(float(np.mean([float(ep["final_box_displacement_m"][name]) for ep in episodes])), 4)
            for name in PUSHABLE_OBSTACLE_STARTS
        },
        "mean_max_box_displacement_m": {
            name: round(float(np.mean([float(ep["max_box_displacement_m"][name]) for ep in episodes])), 4)
            for name in PUSHABLE_OBSTACLE_STARTS
        },
        "robot_contacts_per_episode": round(float(np.mean([int(ep["robot_contacts"]) for ep in episodes])), 4),
        "relocalization_events_per_episode": {
            team: round(mean_team_metric("relocalization_events", team), 4)
            for team in AGENTS
        },
        "abnormal_spin_steps_per_episode": {
            team: round(mean_team_metric("abnormal_spin_steps", team), 4)
            for team in AGENTS
        },
        "static_penetrations_total": sum(int(ep["static_penetrations"][team]) for ep in episodes for team in AGENTS),
        "box_penetrations_total": sum(int(ep["box_penetrations"][team]) for ep in episodes for team in AGENTS),
        "repeat_target_order_events_total": sum(int(ep["repeat_target_order_events"][team]) for ep in episodes for team in AGENTS),
        "simulated_steps_per_second": round(sum(int(ep["steps"]) for ep in episodes) / max(wall_time_s, 1e-9), 1),
    }


def write_csv(episodes: list[dict[str, object]], path: Path):
    fieldnames = [
        "seed",
        "winner",
        "elapsed_s",
        "score_yellow",
        "score_blue",
        "yellow_normal_hits",
        "blue_normal_hits",
        "yellow_hits_before_base",
        "blue_hits_before_base",
        "yellow_push_events",
        "blue_push_events",
        "robot_contacts",
        "yellow_relocalization_events",
        "blue_relocalization_events",
        "yellow_abnormal_spin_steps",
        "blue_abnormal_spin_steps",
        "static_penetrations",
        "box_penetrations",
        "repeat_target_order_events",
        "box_ne_final_displacement_m",
        "box_sw_final_displacement_m",
        "box_ne_max_displacement_m",
        "box_sw_max_displacement_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            writer.writerow(
                {
                    "seed": ep["seed"],
                    "winner": ep["winner"],
                    "elapsed_s": ep["elapsed_s"],
                    "score_yellow": ep["scores"]["yellow"],
                    "score_blue": ep["scores"]["blue"],
                    "yellow_normal_hits": ep["normal_hits"]["yellow"],
                    "blue_normal_hits": ep["normal_hits"]["blue"],
                    "yellow_hits_before_base": ep["normal_hits_before_base"]["yellow"],
                    "blue_hits_before_base": ep["normal_hits_before_base"]["blue"],
                    "yellow_push_events": ep["push_events"]["yellow"],
                    "blue_push_events": ep["push_events"]["blue"],
                    "robot_contacts": ep["robot_contacts"],
                    "yellow_relocalization_events": ep["relocalization_events"]["yellow"],
                    "blue_relocalization_events": ep["relocalization_events"]["blue"],
                    "yellow_abnormal_spin_steps": ep["abnormal_spin_steps"]["yellow"],
                    "blue_abnormal_spin_steps": ep["abnormal_spin_steps"]["blue"],
                    "static_penetrations": sum(ep["static_penetrations"].values()),
                    "box_penetrations": sum(ep["box_penetrations"].values()),
                    "repeat_target_order_events": sum(ep["repeat_target_order_events"].values()),
                    "box_ne_final_displacement_m": ep["final_box_displacement_m"]["box_ne"],
                    "box_sw_final_displacement_m": ep["final_box_displacement_m"]["box_sw"],
                    "box_ne_max_displacement_m": ep["max_box_displacement_m"]["box_ne"],
                    "box_sw_max_displacement_m": ep["max_box_displacement_m"]["box_sw"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the policy against the full strategy contract.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--policy-mode", choices=("auto", "direct", "expert", "residual_expert"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=ROOT / "isaaclab_sim/output/eval/strategy_contract_eval.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "isaaclab_sim/output/eval/strategy_contract_eval.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
    model, checkpoint = load_policy(args.checkpoint, device)
    train_config = checkpoint.get("config", {})
    policy_mode = str(train_config.get("policy_mode", "direct")) if args.policy_mode == "auto" else args.policy_mode
    residual_scale = (
        float(train_config.get("residual_scale", 0.28))
        if args.residual_scale is None
        else float(args.residual_scale)
    )
    started = time.perf_counter()
    episodes = [
        run_episode(
            model,
            seed=args.seed + index,
            max_steps=args.max_steps,
            device=device,
            deterministic=not args.stochastic,
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
        for index in range(args.episodes)
    ]
    wall_time_s = time.perf_counter() - started
    payload = json_safe(
        {
            "summary": summarize(episodes, wall_time_s),
            "checkpoint": str(args.checkpoint),
            "deterministic": not args.stochastic,
            "policy_mode": policy_mode,
            "residual_scale": residual_scale,
            "device": str(device),
            "training_config": train_config,
            "episodes": episodes,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(episodes, args.output_csv)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[INFO] wrote {args.output_json}")
    print(f"[INFO] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
