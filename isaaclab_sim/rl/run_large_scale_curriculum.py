from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import large_scale_50v50_battle as battle


ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_DIR = ROOT / "docs" / "rl_data" / "large_scale_50v50"
CURRICULUM_DIR = ROOT / "docs" / "rl_data" / "large_scale_curriculum"


@dataclass(frozen=True)
class Stage:
    name: str
    agents_per_team: int
    generations: int
    population: int
    episodes_per_candidate: int
    probe_episodes: int
    selection_episodes: int
    eval_episodes: int
    max_steps: int
    base_hp: float
    shield_progress_to_open: float
    capture_rate: float
    contact_limit: float
    min_side_win_rate: float
    min_base_damage: float


STAGES = [
    Stage("stage01_05v05", 5, 80, 14, 4, 10, 48, 128, 560, 8.0, 1.2, 0.105, 22.0, 0.15, 1.0),
    Stage("stage02_10v10", 10, 90, 16, 3, 10, 56, 160, 600, 14.0, 2.2, 0.090, 38.0, 0.20, 2.5),
    Stage("stage03_25v25", 25, 110, 18, 2, 10, 64, 192, 680, 28.0, 5.0, 0.070, 78.0, 0.25, 6.0),
    Stage("stage04_50v50", 50, 150, 20, 2, 10, 72, 256, 760, 45.0, 8.0, 0.060, 135.0, 0.30, 10.0),
]


def ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def common_rule_kwargs(stage: Stage) -> dict[str, Any]:
    return {
        "agents_per_team": stage.agents_per_team,
        "max_steps": stage.max_steps,
        "base_hp": stage.base_hp,
        "base_damage": 1.10,
        "blue_base_damage_multiplier": 1.0,
        "capture_rate": stage.capture_rate,
        "shield_progress_to_open": stage.shield_progress_to_open,
        "contact_radius": None,
        "separation_radius": None,
    }


def train_args(stage: Stage, seed: int, init_checkpoint: str) -> SimpleNamespace:
    return ns(
        seed=seed,
        generations=stage.generations,
        population=stage.population,
        episodes_per_candidate=stage.episodes_per_candidate,
        probe_episodes=stage.probe_episodes,
        elite_frac=0.25,
        sigma=0.52,
        min_sigma=0.07,
        sigma_decay=0.988,
        archive_interval=4,
        archive_size=10,
        init_checkpoint=init_checkpoint,
        log_interval=5,
        selection_episodes=stage.selection_episodes,
        verbose=True,
        **common_rule_kwargs(stage),
    )


def eval_args(stage: Stage, seed: int) -> SimpleNamespace:
    return ns(
        checkpoint=str(BASE_DATA_DIR / "policy_checkpoint.json"),
        episodes=stage.eval_episodes,
        seed=seed,
        **common_rule_kwargs(stage),
    )


def render_args(stage: Stage, seed: int) -> SimpleNamespace:
    return ns(
        checkpoint=str(BASE_DATA_DIR / "policy_checkpoint.json"),
        seed=seed,
        trace_stride=1,
        fps=30,
        seconds=30.0,
        gif_seconds=12.0,
        gif_fps=8,
        width=1920,
        height=1080,
        **common_rule_kwargs(stage),
    )


def stage_passed(stage: Stage, summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if summary["yellow_win_rate"] < stage.min_side_win_rate:
        failures.append(f"yellow win rate {summary['yellow_win_rate']:.3f} < {stage.min_side_win_rate:.3f}")
    if summary["blue_win_rate"] < stage.min_side_win_rate:
        failures.append(f"blue win rate {summary['blue_win_rate']:.3f} < {stage.min_side_win_rate:.3f}")
    if summary["mean_yellow_base_damage"] < stage.min_base_damage:
        failures.append(f"yellow base damage {summary['mean_yellow_base_damage']:.2f} < {stage.min_base_damage:.2f}")
    if summary["mean_blue_base_damage"] < stage.min_base_damage:
        failures.append(f"blue base damage {summary['mean_blue_base_damage']:.2f} < {stage.min_base_damage:.2f}")
    if summary["mean_yellow_base_open_rate"] <= 0.0:
        failures.append("yellow base open rate is zero")
    if summary["mean_blue_base_open_rate"] <= 0.0:
        failures.append("blue base open rate is zero")
    if summary["mean_obstacle_contacts"] > 0.0:
        failures.append(f"obstacle contacts {summary['mean_obstacle_contacts']:.2f} > 0")
    if summary["p95_robot_contacts"] > stage.contact_limit:
        failures.append(f"p95 robot contacts {summary['p95_robot_contacts']:.2f} > {stage.contact_limit:.2f}")
    return not failures, failures


def copy_outputs(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "policy_checkpoint.json",
        "training_summary.json",
        "training_curve.csv",
        "policy_selection.csv",
        "eval_summary.json",
        "eval_episodes.csv",
    ]:
        src = BASE_DATA_DIR / name
        if src.exists():
            shutil.copy2(src, stage_dir / name)


def write_curriculum_summary(rows: list[dict[str, Any]]) -> None:
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    (CURRICULUM_DIR / "curriculum_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (CURRICULUM_DIR / "curriculum_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "stage",
            "agents_per_team",
            "passed",
            "failures",
            "episodes_seen",
            "eval_episodes",
            "yellow_win_rate",
            "blue_win_rate",
            "draw_rate",
            "mean_yellow_base_damage",
            "mean_blue_base_damage",
            "mean_robot_contacts",
            "p95_robot_contacts",
            "mean_obstacle_contacts",
            "checkpoint",
            "wall_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_curriculum(args: argparse.Namespace) -> int:
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    init_checkpoint = ""
    start = time.time()
    for idx, stage in enumerate(STAGES):
        stage_seed = args.seed + idx * 100000
        stage_dir = CURRICULUM_DIR / stage.name
        print(f"[CURRICULUM] start {stage.name}: {stage.agents_per_team}v{stage.agents_per_team}", flush=True)
        ckpt = battle.train(train_args(stage, stage_seed, init_checkpoint))
        eval_payload = battle.evaluate(eval_args(stage, stage_seed + 50000))
        summary = eval_payload["summary"]
        passed, failures = stage_passed(stage, summary)
        copy_outputs(stage_dir)
        checkpoint_path = stage_dir / "policy_checkpoint.json"
        row = {
            "stage": stage.name,
            "agents_per_team": stage.agents_per_team,
            "passed": passed,
            "failures": "; ".join(failures),
            "episodes_seen": ckpt["training"]["episodes_seen"],
            "eval_episodes": summary["episodes"],
            "yellow_win_rate": summary["yellow_win_rate"],
            "blue_win_rate": summary["blue_win_rate"],
            "draw_rate": summary["draw_rate"],
            "mean_yellow_base_damage": summary["mean_yellow_base_damage"],
            "mean_blue_base_damage": summary["mean_blue_base_damage"],
            "mean_robot_contacts": summary["mean_robot_contacts"],
            "p95_robot_contacts": summary["p95_robot_contacts"],
            "mean_obstacle_contacts": summary["mean_obstacle_contacts"],
            "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "wall_time_s": ckpt["training"]["wall_time_s"],
        }
        rows.append(row)
        write_curriculum_summary(rows)
        print(
            f"[CURRICULUM] {stage.name} pass={passed} "
            f"Y/B={summary['yellow_win_rate']:.3f}/{summary['blue_win_rate']:.3f} "
            f"damage={summary['mean_yellow_base_damage']:.1f}/{summary['mean_blue_base_damage']:.1f} "
            f"contacts p95={summary['p95_robot_contacts']:.1f}",
            flush=True,
        )
        if not passed:
            print(f"[CURRICULUM] stage failed; stopping before propagating a bad checkpoint: {'; '.join(failures)}", flush=True)
            return 2
        init_checkpoint = str(checkpoint_path)

    final_stage = STAGES[-1]
    print("[CURRICULUM] final 50v50 render and figures", flush=True)
    battle.render_video(render_args(final_stage, args.seed + 900000))
    battle.make_figures(ns(**common_rule_kwargs(final_stage)))
    battle.write_report()
    rows.append({"stage": "final_artifacts", "agents_per_team": 50, "passed": True, "failures": "", "wall_time_s": time.time() - start})
    write_curriculum_summary(rows)
    print(f"[CURRICULUM] done in {time.time() - start:.1f}s", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curriculum trainer for 5v5 -> 10v10 -> 25v25 -> 50v50 battle policy.")
    parser.add_argument("--seed", type=int, default=607050)
    parser.add_argument("--stop-on-failure", action="store_true", help="Kept for compatibility; stages now always stop on failure.")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_curriculum(build_parser().parse_args()))
