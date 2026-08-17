from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DATA_DIR,
    DEFAULT_THETA,
    config_from_args,
    policy_params
)
from .sim import (
    LargeScaleBattle50v50
)

def side_fitness(metrics: dict[str, Any], side: str) -> float:
    sign = 1.0 if side == "yellow" else -1.0
    score_diff = sign * (metrics["yellow_score"] - metrics["blue_score"])
    if metrics["winner"] == side:
        win_bonus = 35.0
    elif metrics["winner"] == "draw":
        win_bonus = 0.0
    else:
        win_bonus = -35.0
    alive = metrics[f"{side}_alive"]
    base_damage = metrics[f"{side}_base_damage"]
    open_rate = metrics[f"{side}_base_open_rate"]
    shielded = metrics[f"{side}_shielded_base_shots"]
    contacts = metrics["robot_contacts"]
    obstacle = metrics["obstacle_contacts"]
    return float(score_diff + win_bonus + 0.12 * alive + 5.0 * base_damage + 8.0 * open_rate - 0.012 * contacts - 0.015 * obstacle - 0.05 * shielded)


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    winners = [e["winner"] for e in episodes]
    summary = {
        "episodes": n,
        "yellow_win_rate": winners.count("yellow") / n,
        "blue_win_rate": winners.count("blue") / n,
        "draw_rate": winners.count("draw") / n,
    }
    keys = [
        "elapsed_s",
        "yellow_score",
        "blue_score",
        "yellow_alive",
        "blue_alive",
        "yellow_kills",
        "blue_kills",
        "yellow_base_hp",
        "blue_base_hp",
        "robot_contacts",
        "obstacle_contacts",
        "yellow_base_damage",
        "blue_base_damage",
        "yellow_shielded_base_shots",
        "blue_shielded_base_shots",
        "yellow_base_open_rate",
        "blue_base_open_rate",
    ]
    for key in keys:
        summary[f"mean_{key}"] = float(np.mean([e[key] for e in episodes]))
    zone = np.array([e["final_zone_state"] for e in episodes], dtype=np.float64)
    summary["mean_final_zone_state"] = zone.mean(axis=0).round(4).tolist()
    summary["p95_robot_contacts"] = float(np.percentile([e["robot_contacts"] for e in episodes], 95))
    summary["p95_obstacle_contacts"] = float(np.percentile([e["obstacle_contacts"] for e in episodes], 95))
    return summary


def train(args: argparse.Namespace) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    rng = np.random.default_rng(args.seed)
    init_checkpoint = getattr(args, "init_checkpoint", "")
    if init_checkpoint and Path(init_checkpoint).exists():
        theta = np.array(load_checkpoint(Path(init_checkpoint))["theta"], dtype=np.float64)
    else:
        theta = DEFAULT_THETA.copy()
    sigma = float(args.sigma)
    archive: list[np.ndarray] = [DEFAULT_THETA.copy(), theta.copy()]
    curve = []
    best_theta = theta.copy()
    best_fitness = -1e9
    start = __import__("time").time()

    for gen in range(args.generations):
        candidates = []
        fitnesses = []
        for _ in range(args.population):
            candidate = theta + rng.normal(0.0, sigma, size=theta.shape)
            opponent = archive[int(rng.integers(0, len(archive)))]
            scores = []
            for ep in range(args.episodes_per_candidate):
                seed = args.seed + gen * 100000 + ep * 1000 + len(candidates)
                m1 = env.run_episode(candidate, opponent, seed)
                m2 = env.run_episode(opponent, candidate, seed + 17)
                scores.append(side_fitness(m1, "yellow"))
                scores.append(side_fitness(m2, "blue"))
            candidates.append(candidate)
            fitnesses.append(float(np.mean(scores)))
        order = np.argsort(fitnesses)[::-1]
        elite_n = max(2, int(args.population * args.elite_frac))
        elites = np.array([candidates[i] for i in order[:elite_n]])
        elite_scores = np.array([fitnesses[i] for i in order[:elite_n]], dtype=np.float64)
        weights = elite_scores - elite_scores.min() + 1e-6
        weights = weights / weights.sum()
        theta = (elites * weights[:, None]).sum(axis=0)
        gen_best = float(fitnesses[order[0]])
        gen_mean = float(np.mean(fitnesses))
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_theta = candidates[order[0]].copy()
        if gen % max(1, args.archive_interval) == 0:
            archive.append(best_theta.copy())
            archive = archive[-args.archive_size :]
        sigma = max(args.min_sigma, sigma * args.sigma_decay)

        eval_eps = []
        for k in range(args.probe_episodes):
            eval_eps.append(env.run_episode(best_theta, best_theta, args.seed + 900000 + gen * 100 + k))
        probe = summarize_episodes(eval_eps)
        row = {
            "generation": gen,
            "population": args.population,
            "episodes_seen": (gen + 1) * args.population * args.episodes_per_candidate * 2,
            "best_fitness": best_fitness,
            "generation_best_fitness": gen_best,
            "generation_mean_fitness": gen_mean,
            "sigma": sigma,
            "probe_yellow_win_rate": probe["yellow_win_rate"],
            "probe_blue_win_rate": probe["blue_win_rate"],
            "probe_draw_rate": probe["draw_rate"],
            "probe_mean_elapsed_s": probe["mean_elapsed_s"],
            "probe_mean_robot_contacts": probe["mean_robot_contacts"],
            "probe_mean_obstacle_contacts": probe["mean_obstacle_contacts"],
            "probe_mean_yellow_alive": probe["mean_yellow_alive"],
            "probe_mean_blue_alive": probe["mean_blue_alive"],
        }
        curve.append(row)
        with (DATA_DIR / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
            writer.writeheader()
            writer.writerows(curve)
        if args.verbose and (gen == 0 or (gen + 1) % args.log_interval == 0 or gen == args.generations - 1):
            print(
                f"gen {gen + 1:04d}/{args.generations} "
                f"best={best_fitness:.3f} mean={gen_mean:.3f} "
                f"probe Y/B/D={probe['yellow_win_rate']:.2f}/{probe['blue_win_rate']:.2f}/{probe['draw_rate']:.2f} "
                f"contacts={probe['mean_robot_contacts']:.1f}",
                flush=True,
            )

    selection_candidates = [DEFAULT_THETA.copy(), best_theta.copy(), theta.copy()] + [item.copy() for item in archive]
    selection_rows = []
    selected_theta = DEFAULT_THETA.copy()
    selected_score = -1e9
    for idx, candidate in enumerate(selection_candidates):
        eval_eps = [env.run_episode(candidate, candidate, args.seed + 7000000 + idx * 1000 + k) for k in range(args.selection_episodes)]
        summary = summarize_episodes(eval_eps)
        balance_penalty = abs(summary["yellow_win_rate"] - summary["blue_win_rate"])
        base_deficit = max(0.0, 8.0 - summary["mean_yellow_base_damage"]) + max(0.0, 8.0 - summary["mean_blue_base_damage"])
        contact_penalty = max(0.0, summary["mean_robot_contacts"] - 180.0) / 180.0
        score = 20.0 - 16.0 * balance_penalty - 1.4 * base_deficit - 2.0 * contact_penalty
        row = {"candidate": idx, "selection_score": score, **summary}
        selection_rows.append(row)
        if score > selected_score:
            selected_score = score
            selected_theta = candidate.copy()

    training_time = __import__("time").time() - start
    ckpt = {
        "algorithm": "population_based_swarm_flow_policy_search",
        "scenario": f"large_scale_{env.cfg.agents_per_team}v{env.cfg.agents_per_team}_control_zone_base_assault",
        "seed": args.seed,
        "theta": selected_theta.round(8).tolist(),
        "policy_params": policy_params(selected_theta),
        "config": asdict(env.cfg),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "episodes_per_candidate": args.episodes_per_candidate,
            "probe_episodes": args.probe_episodes,
            "episodes_seen": args.generations * args.population * args.episodes_per_candidate * 2,
            "best_fitness": best_fitness,
            "selection_episodes_per_candidate": args.selection_episodes,
            "selected_validation_score": selected_score,
            "wall_time_s": training_time,
        },
    }
    (DATA_DIR / "policy_checkpoint.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    with (DATA_DIR / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    summary = {"checkpoint": "docs/rl_data/large_scale_50v50/policy_checkpoint.json", **ckpt["training"]}
    (DATA_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (DATA_DIR / "policy_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    return ckpt


def load_checkpoint(path: Path = DATA_DIR / "policy_checkpoint.json") -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    ckpt = load_checkpoint(Path(args.checkpoint))
    theta = np.array(ckpt["theta"], dtype=np.float64)
    episodes = [env.run_episode(theta, theta, args.seed + i) for i in range(args.episodes)]
    summary = summarize_episodes(episodes)
    payload = {
        "scenario": f"large_scale_{env.cfg.agents_per_team}v{env.cfg.agents_per_team}_control_zone_base_assault",
        "policy_checkpoint": str(Path(args.checkpoint).as_posix()),
        "summary": summary,
        "episodes": episodes,
    }
    (DATA_DIR / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (DATA_DIR / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "episode",
            "winner",
            "elapsed_s",
            "yellow_score",
            "blue_score",
            "yellow_alive",
            "blue_alive",
            "yellow_kills",
            "blue_kills",
            "yellow_base_hp",
            "blue_base_hp",
            "robot_contacts",
            "obstacle_contacts",
            "yellow_base_damage",
            "blue_base_damage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, ep in enumerate(episodes):
            writer.writerow({"episode": i, **{k: ep[k] for k in fieldnames if k != "episode"}})
    return payload
