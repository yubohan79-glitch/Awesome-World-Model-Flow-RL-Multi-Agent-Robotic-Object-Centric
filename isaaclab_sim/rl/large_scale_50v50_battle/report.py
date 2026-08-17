from __future__ import annotations

import argparse
import csv
import json

import matplotlib.pyplot as plt

from .config import (
    DATA_DIR,
    FIG_DIR,
    ROOT,
    config_from_args
)
from .sim import (
    LargeScaleBattle50v50
)

def make_figures(args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    with (DATA_DIR / "training_curve.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curve_rows.append({k: float(v) for k, v in row.items()})
    eval_payload = json.loads((DATA_DIR / "eval_summary.json").read_text(encoding="utf-8"))
    summary = eval_payload["summary"]

    x = [r["generation"] for r in curve_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=160)
    fig.suptitle("50v50 Swarm Policy Training", fontsize=18, fontweight="bold")
    axes[0, 0].plot(x, [r["best_fitness"] for r in curve_rows], color="#2563eb", lw=2.4, label="best")
    axes[0, 0].plot(x, [r["generation_mean_fitness"] for r in curve_rows], color="#94a3b8", lw=1.8, label="population mean")
    axes[0, 0].set_title("Population fitness")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [r["probe_yellow_win_rate"] for r in curve_rows], color="#eab308", lw=2.2, label="yellow")
    axes[0, 1].plot(x, [r["probe_blue_win_rate"] for r in curve_rows], color="#2563eb", lw=2.2, label="blue")
    axes[0, 1].plot(x, [r["probe_draw_rate"] for r in curve_rows], color="#64748b", lw=1.6, label="draw")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Probe self-play outcome")
    axes[0, 1].legend()
    axes[1, 0].plot(x, [r["probe_mean_robot_contacts"] for r in curve_rows], color="#dc2626", lw=2.2)
    axes[1, 0].set_title("Robot contacts per probe game")
    axes[1, 1].plot(x, [r["probe_mean_yellow_alive"] for r in curve_rows], color="#eab308", lw=2.2, label="yellow")
    axes[1, 1].plot(x, [r["probe_mean_blue_alive"] for r in curve_rows], color="#2563eb", lw=2.2, label="blue")
    axes[1, 1].set_title("Survivors per team")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(True, color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_training.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    labels = ["Yellow win", "Blue win", "Draw", "Y alive", "B alive", "Contacts p95"]
    values = [
        summary["yellow_win_rate"] * 100,
        summary["blue_win_rate"] * 100,
        summary["draw_rate"] * 100,
        summary["mean_yellow_alive"],
        summary["mean_blue_alive"],
        summary["p95_robot_contacts"],
    ]
    colors = ["#eab308", "#2563eb", "#64748b", "#facc15", "#60a5fa", "#dc2626"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("50v50 Evaluation Summary", fontsize=18, fontweight="bold")
    ax.set_ylabel("Percent, count, or p95 event count")
    ax.grid(axis="y", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, f"{value:.1f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_eval.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.suptitle("50v50 Rule-Scoring Closure and Replay Evidence", fontsize=18, fontweight="bold")
    boxes = [
        (4, 22, 18, 10, "Swarm-flow policy\nshared team actor\n50 vehicles/side", "#fef3c7", "#eab308"),
        (28, 22, 18, 10, "Rule simulation\nzones, LOS fire,\nshielded bases", "#dbeafe", "#2563eb"),
        (52, 22, 18, 10, "Scoring closure\nzone control -> shield\nbase damage -> win", "#dcfce7", "#16a34a"),
        (76, 22, 18, 10, "Selection gate\nwin balance,\ncontacts, damage", "#fee2e2", "#dc2626"),
        (16, 5, 24, 9, "256-game evaluation\nY win {0:.1f}% | B win {1:.1f}%\nbase damage {2:.1f}/{3:.1f}".format(
            summary["yellow_win_rate"] * 100,
            summary["blue_win_rate"] * 100,
            summary["mean_yellow_base_damage"],
            summary["mean_blue_base_damage"],
        ), "#f8fafc", "#475569"),
        (58, 5, 28, 9, "IsaacLab replay QA\n100 vehicle-shaped actors\n30 s MP4 + GIF + figures", "#f8fafc", "#475569"),
    ]
    for x0, y0, w, h, text, face, edge in boxes:
        rect = plt.Rectangle((x0, y0), w, h, facecolor=face, edgecolor=edge, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    for start, end in [((22, 27), (28, 27)), ((46, 27), (52, 27)), ((70, 27), (76, 27)), ((85, 22), (72, 14)), ((28, 22), (28, 14)), ((40, 9.5), (58, 9.5))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#334155"})
    ax.text(50, 36, "Promotion requires both tactical behavior and evidence artifacts, not reward-only training.", ha="center", fontsize=11, color="#334155")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_rule_closure.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 50)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8fafc")
    layout_env = LargeScaleBattle50v50(config_from_args(args))
    for rect in layout_env.obstacles:
        xmin, ymin, xmax, ymax = rect
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#cbd5e1", ec="#334155"))
    for i, z in enumerate(layout_env.zones):
        ax.add_patch(plt.Circle(z, 6, fill=False, lw=3, color="#7c3aed"))
        ax.text(z[0], z[1], f"Zone {i+1}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.scatter([4.5], [25], s=800, marker="s", color="#eab308", edgecolor="#0f172a", label="Yellow base")
    ax.scatter([75.5], [25], s=800, marker="s", color="#2563eb", edgecolor="#0f172a", label="Blue base")
    ax.arrow(10, 25, 22, 0, width=0.3, head_width=2.0, color="#eab308", alpha=0.7)
    ax.arrow(70, 25, -22, 0, width=0.3, head_width=2.0, color="#2563eb", alpha=0.7)
    ax.set_title("50v50 Rule Layout: Three Control Zones + Shielded Base Assault", fontsize=18, fontweight="bold")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_rule_layout.png", bbox_inches="tight")
    plt.close(fig)


def write_report() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_payload = json.loads((DATA_DIR / "eval_summary.json").read_text(encoding="utf-8"))
    train_summary = json.loads((DATA_DIR / "training_summary.json").read_text(encoding="utf-8"))
    s = eval_payload["summary"]
    report = f"""# Large-Scale 50v50 Multi-Agent Battle Report

This report documents the first formal large-scale rule-level extension for the repository. It is not a replacement for the two-robot IsaacLab result; it is a new 100-agent benchmark contract used to study scalable multi-agent decision making before expensive full-physics replay.

## Scenario

- Two teams: yellow and blue.
- Agents per team: 50.
- Arena: 80 m x 50 m.
- Objectives: capture three middle control zones, open the enemy base shield, eliminate opponents, then damage the enemy base.
- Obstacles: three static cover/barrier regions.
- Safety metrics: robot contacts, obstacle contacts, shielded base shots, survivors and base health.

## Training

- Algorithm: population-based swarm flow policy search.
- Generations: {train_summary['generations']}.
- Population: {train_summary['population']}.
- Candidate episodes: {train_summary['episodes_per_candidate']}.
- Total training episodes sampled: {train_summary['episodes_seen']}.
- Best fitness: {train_summary['best_fitness']:.4f}.
- Wall time: {train_summary['wall_time_s']:.2f} s.

## Evaluation

- Episodes: {s['episodes']}.
- Yellow win rate: {s['yellow_win_rate'] * 100:.2f}%.
- Blue win rate: {s['blue_win_rate'] * 100:.2f}%.
- Draw rate: {s['draw_rate'] * 100:.2f}%.
- Mean yellow score: {s['mean_yellow_score']:.2f}.
- Mean blue score: {s['mean_blue_score']:.2f}.
- Mean yellow survivors: {s['mean_yellow_alive']:.2f} / 50.
- Mean blue survivors: {s['mean_blue_alive']:.2f} / 50.
- Mean yellow base damage: {s['mean_yellow_base_damage']:.2f}.
- Mean blue base damage: {s['mean_blue_base_damage']:.2f}.
- Mean yellow base open rate: {s['mean_yellow_base_open_rate'] * 100:.2f}%.
- Mean blue base open rate: {s['mean_blue_base_open_rate'] * 100:.2f}%.
- Mean robot contacts: {s['mean_robot_contacts']:.2f}.
- P95 robot contacts: {s['p95_robot_contacts']:.2f}.
- Mean obstacle contacts: {s['mean_obstacle_contacts']:.2f}.
- Mean final zone state: {s['mean_final_zone_state']}.

## Artifacts

- Checkpoint: `docs/rl_data/large_scale_50v50/policy_checkpoint.json`
- Training curve: `docs/rl_data/large_scale_50v50/training_curve.csv`
- Evaluation JSON: `docs/rl_data/large_scale_50v50/eval_summary.json`
- Evaluation CSV: `docs/rl_data/large_scale_50v50/eval_episodes.csv`
- Rule-level preview MP4: `docs/media/large_scale_50v50_replay.mp4`
- Rule-level preview GIF: `docs/media/large_scale_50v50_replay.gif`
- IsaacLab tactical replay MP4: `docs/media/large_scale_50v50_isaaclab_replay.mp4`
- IsaacLab tactical replay GIF: `docs/media/large_scale_50v50_isaaclab_replay.gif`
- Figures: `docs/figures/large_scale_50v50/`

## Boundary

This benchmark validates scalable rule-level 50v50 mechanics and a trained swarm policy baseline. It does not claim IsaacLab rigid-body validation for all 100 robots and does not claim real-robot deployment. Those require a separate physics scaling and Sim2Real evidence package.
"""
    (ROOT / "docs" / "large_scale_50v50_report.md").write_text(report, encoding="utf-8")
