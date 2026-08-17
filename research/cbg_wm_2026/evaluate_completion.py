from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "isaaclab_sim" / "output" / "paper" / "cbg_wm_2026"
VARIANTS = (
    "legacy_sac_flow",
    "no_belief_uncertainty",
    "no_interaction_graph",
    "static_rule_graph",
    "dynamic_graph_no_pairs",
    "full_accgd_cbg_wm",
)
SEEDS = (260707, 260708, 260709)
SCENARIOS = (
    "nominal",
    "held_out_boxes",
    "held_out_target_yaw",
    "delayed_occlusion",
    "low_traction",
    "aggressive_opponent",
)
BASELINES = ("tdmpc2", "dreamerv3", "safedreamer")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def completed(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    status = str(payload.get("status", "")).lower()
    return bool(payload.get("completed") is True or payload.get("pass") is True or status in {"complete", "completed", "success", "passed"})


def nonempty(path: Path, minimum_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= minimum_bytes
    except OSError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_gate() -> tuple[bool, dict[str, Any]]:
    required = (
        "isaaclab_sim/rl/world_model/constraint_graph_dynamics.py",
        "isaaclab_sim/rl/replay/sequence_replay.py",
        "isaaclab_sim/rl/experiments/generate_frozen_datasets.py",
        "isaaclab_sim/rl/experiments/run_paper_suite.py",
        "isaaclab_sim/rl/experiments/evaluate_paper_suite.py",
        "isaaclab_sim/rl/experiments/aggregate_paper_results.py",
        "isaaclab_sim/rl/configs/cbg_wm_paper_suite.yaml",
        "isaaclab_sim/rl/configs/cbg_wm_scenario_splits.yaml",
        "tests/test_constraint_graph_dynamics.py",
        "tests/test_paired_interventions.py",
        "tests/test_paper_statistics.py",
    )
    missing = [item for item in required if not nonempty(ROOT / item)]
    progress = (len(required) - len(missing)) / len(required)
    return not missing, {"required": len(required), "missing": missing, "progress": progress}


def tests_gate() -> tuple[bool, dict[str, Any]]:
    path = OUTPUT / "validation" / "unit_tests.json"
    payload = load_json(path)
    required = {
        "tests/test_cbg_world_model.py",
        "tests/test_rl_strategy_contract.py",
        "tests/test_constraint_graph_dynamics.py",
        "tests/test_paired_interventions.py",
        "tests/test_paper_statistics.py",
    }
    recorded = {str(value).replace("\\", "/") for value in (payload or {}).get("test_files", ())}
    covered = len(required & recorded)
    command_passed = completed(payload) and int((payload or {}).get("exit_code", 1)) == 0
    progress = covered / len(required) if command_passed else 0.0
    ok = command_passed and covered == len(required)
    return ok, {"path": str(path), "payload": payload, "required": sorted(required), "progress": progress}


def pilots_gate() -> tuple[bool, dict[str, Any]]:
    missing = []
    for variant in ("static_rule_graph", "full_accgd_cbg_wm"):
        payload = load_json(OUTPUT / "pilot" / variant / "status.json")
        if not completed(payload) or int((payload or {}).get("environment_steps", 0)) < 20000:
            missing.append(variant)
    return not missing, {"missing": missing, "progress": (2 - len(missing)) / 2}


def training_gate() -> tuple[bool, dict[str, Any]]:
    missing: list[str] = []
    hashes: list[str] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run = OUTPUT / "train" / variant / f"seed_{seed}"
            status = load_json(run / "exit_status.json")
            checkpoint = run / "checkpoint_best.pt"
            checksum = run / "checkpoint.sha256"
            manifest = load_json(run / "manifest.json")
            if not completed(status) or not nonempty(checkpoint, 1024) or not nonempty(checksum) or not manifest:
                missing.append(f"{variant}:{seed}")
                continue
            expected = checksum.read_text(encoding="ascii").strip().split()[0].lower()
            actual = sha256(checkpoint)
            if expected != actual:
                missing.append(f"{variant}:{seed}:checksum")
                continue
            hashes.append(actual)
    unique = len(hashes) == len(VARIANTS) * len(SEEDS) and len(set(hashes)) == len(hashes)
    progress = len(hashes) / 18
    return not missing and unique, {"complete": len(hashes), "expected": 18, "unique": unique, "missing": missing, "progress": progress}


def frozen_data_gate() -> tuple[bool, dict[str, Any]]:
    manifest = load_json(OUTPUT / "frozen_data" / "manifest.json")
    audit = load_json(OUTPUT / "frozen_data" / "split_audit.json")
    missing = []
    for scenario in SCENARIOS:
        folder = OUTPUT / "frozen_data" / scenario
        if not nonempty(folder / "prediction.npz", 1024):
            missing.append(f"{scenario}:prediction")
        if not nonempty(folder / "interventions.npz", 1024):
            missing.append(f"{scenario}:interventions")
    hashes_ok = bool(manifest and manifest.get("split_sha256") and audit and audit.get("overlap_count") == 0)
    file_progress = (12 - len(missing)) / 12
    progress = (12 * file_progress + int(hashes_ok) + int(completed(audit))) / 14
    return not missing and completed(audit) and hashes_ok, {"missing": missing, "hashes_ok": hashes_ok, "audit": audit, "progress": progress}


def evaluation_gate() -> tuple[bool, dict[str, Any]]:
    missing: list[str] = []
    complete_count = 0
    for variant in VARIANTS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                cell = OUTPUT / "eval" / variant / f"seed_{seed}" / scenario
                summary = load_json(cell / "summary.json")
                raw = cell / "episodes.parquet"
                if not raw.exists():
                    raw = cell / "episodes.csv"
                if completed(summary) and int((summary or {}).get("match_count", 0)) == 256 and nonempty(raw, 256):
                    complete_count += 1
                else:
                    missing.append(f"{variant}:{seed}:{scenario}")
    return complete_count == 108, {"complete": complete_count, "expected": 108, "missing": missing, "progress": complete_count / 108}


def aggregate_gate() -> tuple[bool, dict[str, Any]]:
    required = (
        "tables/table1_win_risk.csv",
        "tables/table2_prediction_edges.csv",
        "tables/table3_calibration_cvar.csv",
        "statistics/primary_hypotheses.json",
        "statistics/hierarchical_bootstrap.json",
        "statistics/holm_bonferroni.json",
        "figures/win_risk_pareto.png",
        "figures/reliability_diagrams.png",
        "figures/intervention_effects.png",
    )
    missing = [item for item in required if not nonempty(OUTPUT / "aggregate" / item, 128)]
    hypotheses = load_json(OUTPUT / "aggregate" / "statistics" / "primary_hypotheses.json")
    progress = (len(required) - len(missing)) / len(required)
    return not missing and completed(hypotheses), {"missing": missing, "hypotheses": hypotheses, "progress": progress}


def external_gate() -> tuple[bool, dict[str, Any]]:
    missing = []
    complete_count = 0
    for baseline in BASELINES:
        for seed in SEEDS:
            run = OUTPUT / "external" / baseline / f"seed_{seed}"
            if not completed(load_json(run / "exit_status.json")) or not nonempty(run / "checkpoint_best.pt", 1024):
                missing.append(f"{baseline}:{seed}")
            else:
                complete_count += 1
    return not missing, {"missing": missing, "complete": complete_count, "expected": 9, "progress": complete_count / 9}


def public_benchmark_gate() -> tuple[bool, dict[str, Any]]:
    summary = load_json(OUTPUT / "public_benchmark" / "summary.json")
    tasks = tuple((summary or {}).get("tasks", ()))
    methods = tuple((summary or {}).get("methods", ()))
    ok = completed(summary) and len(tasks) >= 3 and all(name in methods for name in ("static_rule_graph", "dynamic_graph_no_pairs", "full_accgd_cbg_wm", "tdmpc2", "dreamerv3"))
    return ok, {"summary": summary, "progress": float(ok)}


def hardware_gate() -> tuple[bool, dict[str, Any]]:
    summary = load_json(OUTPUT / "hardware" / "paired_safety_latency.json")
    ok = completed(summary) and int((summary or {}).get("trial_count", 0)) > 0 and bool((summary or {}).get("raw_log_paths"))
    return ok, {"summary": summary, "progress": float(ok)}


def reproducibility_gate() -> tuple[bool, dict[str, Any]]:
    manifest = load_json(OUTPUT / "aggregate" / "reproducibility_manifest.json")
    ok = completed(manifest) and bool((manifest or {}).get("split_sha256")) and bool((manifest or {}).get("worktree_diff_sha256"))
    return ok, {"manifest": manifest, "progress": float(ok)}


def report_gate() -> tuple[bool, dict[str, Any]]:
    report = OUTPUT / "aggregate" / "final_experiment_report.md"
    acceptance = load_json(OUTPUT / "aggregate" / "acceptance_checklist.json")
    ok = nonempty(report, 1024) and completed(acceptance) and bool((acceptance or {}).get("all_required_artifacts_verified"))
    return ok, {"report": str(report), "acceptance": acceptance, "progress": float(ok)}


def main() -> None:
    checks = {
        "engineering": code_gate(),
        "unit_tests": tests_gate(),
        "pilots": pilots_gate(),
        "core_training": training_gate(),
        "frozen_data": frozen_data_gate(),
        "evaluation_108_cells": evaluation_gate(),
        "aggregate_statistics": aggregate_gate(),
        "external_baselines": external_gate(),
        "public_benchmark": public_benchmark_gate(),
        "hardware_trials": hardware_gate(),
        "reproducibility": reproducibility_gate(),
        "final_report": report_gate(),
    }
    passed = sum(int(value[0]) for value in checks.values())
    score = sum(float(value[1].get("progress", float(value[0]))) for value in checks.values()) / len(checks)
    result = {
        "pass": passed == len(checks),
        "score": score,
        "passed_gates": passed,
        "total_gates": len(checks),
        "details": {name: {"pass": value[0], **value[1]} for name, value in checks.items()},
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
