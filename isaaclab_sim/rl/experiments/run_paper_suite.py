from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = RL_ROOT / "configs" / "cbg_wm_paper_suite.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip()


def diff_sha256() -> str:
    digest = hashlib.sha256()
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True
    )
    digest.update(tracked.stdout)
    untracked = git_value("ls-files", "--others", "--exclude-standard").splitlines()
    for relative in sorted(untracked):
        path = REPO_ROOT / relative
        if path.is_file():
            digest.update(relative.replace("\\", "/").encode("utf-8"))
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def cli_args(values: dict[str, object]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            result.extend([flag, str(value)])
    return result


def load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_completed_run(run_dir: Path, timesteps: int, seed: int, variant: str) -> bool:
    status = load_json(run_dir / "exit_status.json")
    manifest = load_json(run_dir / "manifest.json")
    checkpoint = run_dir / "checkpoint_best.pt"
    checksum = run_dir / "checkpoint.sha256"
    if not status or not manifest or not checkpoint.is_file() or not checksum.is_file():
        return False
    if status.get("completed") is not True or int(status.get("environment_steps", 0)) != timesteps:
        return False
    if int(manifest.get("seed", -1)) != seed or manifest.get("training_variant") != variant:
        return False
    return checksum.read_text(encoding="ascii").strip().split()[0] == sha256(checkpoint)


def run_one(
    *,
    variant: str,
    seed: int,
    timesteps: int,
    run_dir: Path,
    base_config: Path,
    variant_config: dict[str, object],
    device: str,
    dry_run: bool,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RL_ROOT / "train_world_model_sacflow_selfplay.py"),
        "--config",
        str(base_config),
        "--training-variant",
        variant,
        "--seed",
        str(seed),
        "--timesteps",
        str(timesteps),
        "--device",
        device,
        "--output",
        str(run_dir),
        *cli_args(variant_config),
    ]
    if dry_run:
        print(json.dumps({"variant": variant, "seed": seed, "command": command}))
        return 0
    started = datetime.now(timezone.utc)
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finished = datetime.now(timezone.utc)
    summary = load_json(run_dir / "training_summary.json")
    policy = run_dir / "policy.pt"
    completed = (
        result.returncode == 0
        and summary is not None
        and int(summary.get("config", {}).get("timesteps", -1)) == timesteps
        and int(summary.get("config", {}).get("seed", -1)) == seed
        and summary.get("config", {}).get("training_variant") == variant
        and policy.is_file()
    )
    if completed:
        checkpoint = run_dir / "checkpoint_best.pt"
        shutil.copy2(policy, checkpoint)
        checksum = sha256(checkpoint)
        (run_dir / "checkpoint.sha256").write_text(checksum + "  checkpoint_best.pt\n", encoding="ascii")
        manifest = {
            "status": "completed",
            "completed": True,
            "training_variant": variant,
            "seed": seed,
            "environment_steps": timesteps,
            "checkpoint_sha256": checksum,
            "git_head": git_value("rev-parse", "HEAD"),
            "worktree_diff_sha256": diff_sha256(),
            "base_config_sha256": sha256(base_config),
            "command": command,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = {
        "status": "completed" if completed else "failed",
        "completed": completed,
        "exit_code": result.returncode,
        "environment_steps": timesteps if completed else int((summary or {}).get("config", {}).get("timesteps", 0)),
        "training_variant": variant,
        "seed": seed,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
    }
    (run_dir / "exit_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0 if completed else 1


def resolve_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RL_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-safe orchestrator for the registered 18-run CBG-WM suite.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("pilot", "train", "all"), default="all")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--timesteps-override", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    suite = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ablation_path = resolve_config_path(str(suite["ablation_config"]))
    base_config = resolve_config_path(str(suite["base_config"]))
    ablations = yaml.safe_load(ablation_path.read_text(encoding="utf-8"))
    output_root = resolve_config_path(str(suite["output_root"]))
    variants = list(args.variants or ablations["variants"].keys())
    unknown = sorted(set(variants) - set(ablations["variants"]))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    failures = 0
    phases = ("pilot", "train") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "pilot":
            phase_variants = [value for value in suite["pilots"]["variants"] if value in variants]
            seeds = [int((args.seeds or suite["seeds"])[0])]
            timesteps = int(args.timesteps_override or suite["pilots"]["timesteps"])
        else:
            phase_variants = variants
            seeds = [int(value) for value in (args.seeds or suite["seeds"])]
            timesteps = int(args.timesteps_override or suite["timesteps"])
        for variant in phase_variants:
            for seed in seeds:
                run_dir = (
                    output_root / "pilot" / variant
                    if phase == "pilot"
                    else output_root / "train" / variant / f"seed_{seed}"
                )
                if not args.no_resume and valid_completed_run(run_dir, timesteps, seed, variant):
                    print(f"[SKIP] {phase} {variant} seed={seed} already validated", flush=True)
                    continue
                result = run_one(
                    variant=variant,
                    seed=seed,
                    timesteps=timesteps,
                    run_dir=run_dir,
                    base_config=base_config,
                    variant_config=dict(ablations["variants"][variant]),
                    device=args.device,
                    dry_run=args.dry_run,
                )
                if result != 0:
                    failures += 1
                    if not args.dry_run:
                        print(f"[FAIL] {phase} {variant} seed={seed}; see {run_dir / 'train.log'}", flush=True)
                        return 1
                elif phase == "pilot" and not args.dry_run:
                    source = run_dir / "exit_status.json"
                    status = load_json(source) or {}
                    status["environment_steps"] = timesteps
                    source.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
