from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from experiments.paper_statistics import (
    fixed_tail_cvar,
    hierarchical_bootstrap_ci,
    holm_bonferroni,
    paired_permutation_pvalue,
)
from experiments.scenario_protocol import SCENARIOS


VARIANTS = (
    "legacy_sac_flow",
    "no_belief_uncertainty",
    "no_interaction_graph",
    "static_rule_graph",
    "dynamic_graph_no_pairs",
    "full_accgd_cbg_wm",
)
SEEDS = (260707, 260708, 260709)
RISK_NAMES = ("collision", "penetration", "illegal_fire", "los_or_range")


def load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_cells(root: Path) -> tuple[dict[tuple[str, int, str], dict[str, object]], dict[tuple[str, int, str], list[dict[str, str]]], list[str]]:
    summaries = {}
    episodes = {}
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                folder = root / "eval" / variant / f"seed_{seed}" / scenario
                summary = load_json(folder / "summary.json")
                raw_path = folder / "episodes.csv"
                if not summary or summary.get("completed") is not True or int(summary.get("match_count", 0)) != 256 or not raw_path.is_file():
                    missing.append(f"{variant}:{seed}:{scenario}")
                    continue
                key = (variant, seed, scenario)
                summaries[key] = summary
                episodes[key] = read_csv(raw_path)
    return summaries, episodes, missing


def win_value(row: dict[str, str]) -> float:
    return 1.0 if row["winner"] == row["ego_team"] else 0.5 if row["winner"] == "draw" else 0.0


def paired_cell_values(
    episodes: dict[tuple[str, int, str], list[dict[str, str]]],
    left: str,
    right: str,
    metric,
) -> np.ndarray:
    differences = []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            left_rows = episodes[(left, seed, scenario)]
            right_rows = episodes[(right, seed, scenario)]
            key = lambda row: (row["world_seed"], row["opponent"], row["ego_team"])
            left_map = {key(row): metric(row) for row in left_rows}
            right_map = {key(row): metric(row) for row in right_rows}
            shared = sorted(left_map.keys() & right_map.keys())
            differences.extend(left_map[item] - right_map[item] for item in shared)
    return np.asarray(differences, dtype=np.float64)


def build_tables(root: Path, summaries, episodes) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    win_rows = []
    prediction_rows = []
    calibration_rows = []
    for key, summary in summaries.items():
        variant, seed, scenario = key
        rows = episodes[key]
        costs = np.asarray([[float(row[f"cost_{name}"]) for name in RISK_NAMES] for row in rows])
        win_rows.append({
            "variant": variant,
            "seed": seed,
            "scenario": scenario,
            "matches": len(rows),
            "win_score": float(np.mean([win_value(row) for row in rows])),
            "win_rate": float(np.mean([row["winner"] == row["ego_team"] for row in rows])),
            "draw_rate": float(np.mean([row["winner"] == "draw" for row in rows])),
            "cvar_total_0_90": fixed_tail_cvar(costs.sum(axis=1), 0.90),
            "shield_intervention_rate": float(np.mean([float(row["shield_intervention_rate"]) for row in rows])),
        })
        prediction = summary.get("prediction", {})
        if prediction.get("status") == "completed":
            for horizon, values in prediction["horizons"].items():
                prediction_rows.append({
                    "variant": variant,
                    "seed": seed,
                    "scenario": scenario,
                    "horizon": int(horizon),
                    "physical_rmse": values["physical_rmse"],
                    "position_rmse_normalized": values["position_rmse_normalized"],
                    "edge_presence_macro_f1": prediction["edge_presence"]["macro_f1"],
                    "edge_event_macro_f1": prediction["edge_events"]["macro_f1"],
                    "event_time_mae_steps": prediction["event_time_mae_steps"],
                    "duration_macro_f1": prediction["duration_bucket"]["macro_f1"],
                })
            for risk_name, values in prediction["risk_calibration"].items():
                calibration_rows.append({
                    "variant": variant,
                    "seed": seed,
                    "scenario": scenario,
                    "risk": risk_name,
                    "brier": values["brier"],
                    "binary_nll": values["binary_nll"],
                    "ece_15_equal_mass": values["ece_15_equal_mass"],
                    "positive_transitions": values["positive_transitions"],
                    "predicted_cvar": prediction["cvar_0_90"][risk_name]["predicted"],
                    "realized_cvar": prediction["cvar_0_90"][risk_name]["realized"],
                    "cvar_absolute_error": prediction["cvar_0_90"][risk_name]["absolute_error"],
                })
    return win_rows, prediction_rows, calibration_rows


def create_figures(output: Path, win_rows, calibration_rows, summaries) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for variant in VARIANTS:
        rows = [row for row in win_rows if row["variant"] == variant]
        if rows:
            ax.scatter(
                np.mean([row["cvar_total_0_90"] for row in rows]),
                np.mean([row["win_score"] for row in rows]),
                label=variant,
            )
    ax.set_xlabel("Realized total cost CVaR 0.90")
    ax.set_ylabel("Win score")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "win_risk_pareto.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    for axis, risk in zip(axes.flat, RISK_NAMES):
        rows = [row for row in calibration_rows if row["risk"] == risk]
        axis.bar(range(len(VARIANTS)), [np.mean([row["ece_15_equal_mass"] for row in rows if row["variant"] == variant]) for variant in VARIANTS])
        axis.set_title(risk)
        axis.set_xticks([])
        axis.set_ylabel("ECE")
    fig.tight_layout()
    fig.savefig(figures / "reliability_diagrams.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    mechanisms = ("push_box", "remove_armor")
    width = 0.12
    for variant_index, variant in enumerate(VARIANTS[1:]):
        values = []
        for mechanism in mechanisms:
            selected = []
            for (cell_variant, _seed, _scenario), summary in summaries.items():
                payload = summary.get("interventions", {})
                if cell_variant == variant and payload.get("status") == "completed":
                    selected.append(payload["mechanisms"][mechanism]["action_selection_accuracy"])
            values.append(float(np.mean(selected)))
        ax.bar(np.arange(2) + variant_index * width, values, width=width, label=variant)
    ax.set_xticks(np.arange(2) + 2 * width, mechanisms)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Intervention choice accuracy")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "intervention_effects.png", dpi=180)
    plt.close(fig)


def worktree_diff_sha256() -> str:
    result = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True)
    digest = hashlib.sha256(result.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    for relative in sorted(untracked.stdout.splitlines()):
        path = REPO_ROOT / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def external_complete(root: Path) -> bool:
    for method in ("tdmpc2", "dreamerv3", "safedreamer"):
        for seed in SEEDS:
            status = load_json(root / "external" / method / f"seed_{seed}" / "exit_status.json")
            if not status or status.get("completed") is not True:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate verified CBG-WM paper-suite artifacts.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "isaaclab_sim" / "output" / "paper" / "cbg_wm_2026")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "aggregate"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "statistics").mkdir(parents=True, exist_ok=True)
    summaries, episodes, missing = collect_cells(root)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"cannot aggregate: {len(missing)} of 108 evaluation cells are incomplete")
    if not summaries:
        raise RuntimeError("no completed evaluation cells found")
    win_rows, prediction_rows, calibration_rows = build_tables(root, summaries, episodes)
    write_csv(output / "tables" / "table1_win_risk.csv", win_rows)
    if prediction_rows:
        write_csv(output / "tables" / "table2_prediction_edges.csv", prediction_rows)
    if calibration_rows:
        write_csv(output / "tables" / "table3_calibration_cvar.csv", calibration_rows)

    complete_matrix = not missing
    hypotheses: dict[str, object] = {
        "status": "completed" if complete_matrix else "incomplete",
        "completed": complete_matrix,
        "confirmatory_scope": "three training seeds support system-level uncertainty only; algorithm-level significance requires Stage B with >=10 seeds",
        "missing_cells": missing,
        "tests": {},
    }
    pvalues = {}
    if complete_matrix:
        win_t5_t3 = paired_cell_values(episodes, "full_accgd_cbg_wm", "static_rule_graph", win_value)
        risk_t3_t5 = paired_cell_values(
            episodes,
            "static_rule_graph",
            "full_accgd_cbg_wm",
            lambda row: sum(float(row[f"cost_{name}"]) for name in RISK_NAMES),
        )
        pvalues["win_t5_gt_t3"] = paired_permutation_pvalue(win_t5_t3, seed=31)
        pvalues["risk_t5_lt_t3"] = paired_permutation_pvalue(risk_t3_t5, seed=32)
        hypotheses["tests"] = {
            "win_t5_gt_t3": {"mean_paired_difference": float(win_t5_t3.mean()), "p_value": pvalues["win_t5_gt_t3"]},
            "risk_t5_lt_t3": {"mean_paired_difference": float(risk_t3_t5.mean()), "p_value": pvalues["risk_t5_lt_t3"]},
        }
    (output / "statistics" / "primary_hypotheses.json").write_text(json.dumps(hypotheses, indent=2), encoding="utf-8")
    holm = {"status": "completed" if complete_matrix else "incomplete", "completed": complete_matrix, "results": holm_bonferroni(pvalues) if pvalues else {}}
    (output / "statistics" / "holm_bonferroni.json").write_text(json.dumps(holm, indent=2), encoding="utf-8")

    values = np.asarray([row["win_score"] for row in win_rows], dtype=np.float64)
    seeds = np.asarray([row["seed"] for row in win_rows], dtype=np.int64)
    blocks = np.asarray([SCENARIOS.index(row["scenario"]) for row in win_rows], dtype=np.int64)
    bootstrap = {
        "status": "completed" if complete_matrix else "incomplete",
        "completed": complete_matrix,
        "overall_win_score": hierarchical_bootstrap_ci(values, seeds, blocks, samples=10_000, seed=2026),
    }
    (output / "statistics" / "hierarchical_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    create_figures(output, win_rows, calibration_rows, summaries)

    frozen_manifest = load_json(root / "frozen_data" / "manifest.json") or {}
    reproducibility = {
        "status": "completed" if complete_matrix and bool(frozen_manifest.get("split_sha256")) else "incomplete",
        "completed": complete_matrix and bool(frozen_manifest.get("split_sha256")),
        "split_sha256": frozen_manifest.get("split_sha256"),
        "worktree_diff_sha256": worktree_diff_sha256(),
        "training_seeds": list(SEEDS),
        "evaluation_cells": len(summaries),
        "matches": sum(len(rows) for rows in episodes.values()),
    }
    (output / "reproducibility_manifest.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")

    public = load_json(root / "public_benchmark" / "summary.json") or {}
    hardware = load_json(root / "hardware" / "paired_safety_latency.json") or {}
    all_verified = bool(
        complete_matrix
        and external_complete(root)
        and public.get("completed") is True
        and hardware.get("completed") is True
        and int(hardware.get("trial_count", 0)) > 0
    )
    acceptance = {
        "status": "completed" if all_verified else "incomplete",
        "completed": all_verified,
        "all_required_artifacts_verified": all_verified,
        "evaluation_cells": len(summaries),
        "external_baselines_complete": external_complete(root),
        "public_benchmark_complete": public.get("completed") is True,
        "hardware_trials_complete": hardware.get("completed") is True and int(hardware.get("trial_count", 0)) > 0,
        "missing_evaluation_cells": missing,
    }
    (output / "acceptance_checklist.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    report = f"""# CBG-WM Formal Experiment Report

This report is generated only from mechanically validated artifacts under `{root}`.

## Completion

- Internal training variants: {len(VARIANTS)} variants x {len(SEEDS)} seeds.
- Completed checkpoint-scenario cells: {len(summaries)} / 108.
- Completed online matches: {sum(len(rows) for rows in episodes.values())} / 27648.
- External baselines complete: {acceptance['external_baselines_complete']}.
- Public benchmark complete: {acceptance['public_benchmark_complete']}.
- Hardware trials complete: {acceptance['hardware_trials_complete']}.
- All required artifacts verified: {all_verified}.

## Statistical Scope

The three registered training seeds quantify system-level variation but do not support a broad algorithm-level significance claim. Any wording of statistical significance requires the preregistered Stage B confirmation with at least ten training seeds. Holm correction, paired permutation results, and hierarchical bootstrap intervals are saved under `statistics/`.

## Method Claim

The supported claim is action-conditioned constraint-graph lifecycle prediction and controlled interventional consistency under partial observability. The experiment does not claim identifiable causal structure. T5 must improve edge-change prediction and intervention decision quality relative to T3/T4; a win-rate-only gain is insufficient.

## Artifact Policy

Prediction data, paired interventions, raw episode rows, checkpoint checksums, split hashes, and worktree hashes remain available for independent recomputation. Missing hardware or public benchmark evidence is reported as incomplete and is never replaced with synthetic logs.
"""
    (output / "final_experiment_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(acceptance, indent=2))
    return 0 if all_verified or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
