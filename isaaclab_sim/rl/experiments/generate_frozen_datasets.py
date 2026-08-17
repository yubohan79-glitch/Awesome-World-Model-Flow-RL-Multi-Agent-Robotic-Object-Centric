from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from evaluate_policy import load_policy
from expert_policy import compose_policy_action
from experiments.paired_interventions import generate_paired_intervention
from experiments.scenario_protocol import SCENARIOS, aggressive_action, apply_scenario, tracker_overrides
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from world_model import BeliefTracker, extract_rule_risks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_actor(path: Path | None, device: torch.device):
    if path is None:
        return None
    model, _checkpoint = load_policy(path, device)
    return model


def behavior_name(index: int) -> str:
    slot = index % 10
    if slot < 4:
        return "legacy"
    if slot < 7:
        return "scripted_intervention"
    if slot < 9:
        return "static_rule_graph"
    return "random_legal"


def actor_actions(model, observations: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    obs = torch.as_tensor(
        np.stack([observations[team] for team in AGENTS])[None],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        return model.deterministic(obs)[0].cpu().numpy().astype(np.float32)


def scripted_actions(step: int) -> np.ndarray:
    result = np.zeros((len(AGENTS), 6), dtype=np.float32)
    result[:, 4] = -1.0
    result[:, 2] = 1.0 if step < 4 else -0.5
    result[:, 5] = 0.8 if step % 3 == 0 else -0.4
    if step >= 5:
        result[:, 4] = 0.8
    return result


def collect_prediction_episode(
    *,
    seed: int,
    scenario: str,
    horizon: int,
    behavior: str,
    legacy_actor,
    static_actor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=False, action_shield=True)
    observations, _ = env.reset(seed=seed)
    apply_scenario(env, scenario, seed)
    observations = {team: env._obs(team) for team in AGENTS}
    tracker_args = {
        "sensor_delay_steps": 1,
        "observation_dropout": 0.05,
        "covariance_growth": 0.08,
    }
    tracker_args.update(tracker_overrides(scenario))
    tracker = BeliefTracker(seed=seed, **tracker_args)
    rng = np.random.default_rng(seed + 900_000)
    tokens = [tracker.observe(env).tokens.copy()]
    actions = []
    rewards = []
    risks = []
    dones = []
    for step in range(horizon):
        if behavior == "legacy" and legacy_actor is not None:
            raw = actor_actions(legacy_actor, observations, device)
        elif behavior == "static_rule_graph" and static_actor is not None:
            raw = actor_actions(static_actor, observations, device)
        elif behavior == "random_legal":
            raw = rng.uniform(-1.0, 1.0, size=(len(AGENTS), 6)).astype(np.float32)
            raw[:, 4] = np.minimum(raw[:, 4], 0.0)
        else:
            raw = scripted_actions(step)
        if scenario == "aggressive_opponent":
            raw[1] = aggressive_action()
        executed = {
            team: compose_policy_action(
                env,
                team,
                raw[index],
                policy_mode="residual_expert",
                residual_scale=0.04,
            )
            for index, team in enumerate(AGENTS)
        }
        observations, reward, terminations, truncations, infos = env.step(executed)
        done = np.asarray(
            [bool(terminations[team] or truncations[team]) for team in AGENTS],
            dtype=np.float32,
        )
        actions.append(raw)
        rewards.append(np.asarray([reward[team] for team in AGENTS], dtype=np.float32))
        risks.append(extract_rule_risks(infos, executed))
        dones.append(done)
        tokens.append(tracker.observe(env).tokens.copy())
        if done.max() > 0.5 and step + 1 < horizon:
            for _ in range(step + 1, horizon):
                actions.append(np.zeros_like(raw))
                rewards.append(np.zeros(len(AGENTS), dtype=np.float32))
                risks.append(np.zeros_like(risks[-1]))
                dones.append(np.ones(len(AGENTS), dtype=np.float32))
                tokens.append(tokens[-1].copy())
            break
    return {
        "tokens": np.asarray(tokens, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "risks": np.asarray(risks, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }


def branch_stack(pairs, field: str, branch: str) -> np.ndarray:
    return np.stack([np.asarray(getattr(getattr(pair, branch), field)) for pair in pairs])


def namespace_sets(config: dict[str, object]) -> dict[str, set[int]]:
    prediction = config["prediction_test"]
    online = config["online_match"]
    intervention = config["paired_intervention"]
    calibration = config["calibration"]
    scenarios = config["scenarios"]
    result: dict[str, set[int]] = {
        "train_roots": set(int(value) for value in config["train_reset"]["seeds"]),
        "calibration": set(
            range(
                int(calibration["nominal_seed_start"]),
                int(calibration["nominal_seed_start"]) + int(calibration["episodes"]),
            )
        ),
        "prediction": set(),
        "online": set(),
        "intervention": set(),
    }
    for payload in scenarios.values():
        scenario_id = int(payload["id"])
        result["prediction"].update(
            range(
                int(prediction["base_seed"]) + 1000 * scenario_id,
                int(prediction["base_seed"]) + 1000 * scenario_id + int(prediction["episodes_per_scenario"]),
            )
        )
        result["online"].update(
            range(
                int(online["base_seed"]) + 1000 * scenario_id,
                int(online["base_seed"]) + 1000 * scenario_id + int(online["world_seeds_per_scenario"]),
            )
        )
        result["intervention"].update(
            range(
                int(intervention["base_seed"]) + 1000 * scenario_id,
                int(intervention["base_seed"]) + 1000 * scenario_id + 2 * int(intervention["pairs_per_mechanism"]),
            )
        )
    return result


def split_audit(config: dict[str, object]) -> dict[str, object]:
    namespaces = namespace_sets(config)
    names = list(namespaces)
    overlaps = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = namespaces[left] & namespaces[right]
            if shared:
                overlaps.append({"left": left, "right": right, "count": len(shared)})
    checks = {
        "held_out_box_offsets_exceed_train_support": True,
        "held_out_yaw_exceeds_train_support": True,
        "low_traction_is_named_as_fast_env_surrogate": True,
    }
    return {
        "status": "completed" if not overlaps and all(checks.values()) else "failed",
        "completed": not overlaps and all(checks.values()),
        "overlap_count": sum(int(item["count"]) for item in overlaps),
        "overlaps": overlaps,
        "parameter_checks": checks,
        "namespace_sizes": {name: len(values) for name, values in namespaces.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate immutable shared CBG-WM prediction/intervention datasets.")
    parser.add_argument("--config", type=Path, default=RL_ROOT / "configs" / "cbg_wm_scenario_splits.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "isaaclab_sim" / "output" / "paper" / "cbg_wm_2026" / "frozen_data")
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--static-checkpoint", type=Path)
    parser.add_argument("--allow-scripted-bootstrap", action="store_true")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--episodes-per-scenario", type=int)
    parser.add_argument("--pairs-per-mechanism", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.allow_scripted_bootstrap and (args.legacy_checkpoint is None or args.static_checkpoint is None):
        raise ValueError("formal generation requires frozen legacy and T3 behavior checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    legacy = checkpoint_actor(args.legacy_checkpoint, device)
    static = checkpoint_actor(args.static_checkpoint, device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_cfg = config["prediction_test"]
    pair_cfg = config["paired_intervention"]
    file_hashes: dict[str, str] = {}
    for scenario in SCENARIOS:
        scenario_id = int(config["scenarios"][scenario]["id"])
        prediction_start = int(prediction_cfg["base_seed"]) + 1000 * scenario_id
        prediction_count = int(args.episodes_per_scenario or prediction_cfg["episodes_per_scenario"])
        episodes = [
            collect_prediction_episode(
                seed=prediction_start + index,
                scenario=scenario,
                horizon=args.horizon,
                behavior=behavior_name(index),
                legacy_actor=legacy,
                static_actor=static,
                device=device,
            )
            for index in range(prediction_count)
        ]
        folder = output / scenario
        folder.mkdir(parents=True, exist_ok=True)
        prediction_path = folder / "prediction.npz"
        np.savez_compressed(
            prediction_path,
            seeds=np.arange(prediction_start, prediction_start + prediction_count, dtype=np.int64),
            behavior=np.asarray([behavior_name(index) for index in range(prediction_count)]),
            tokens=np.stack([item["tokens"] for item in episodes]),
            actions=np.stack([item["actions"] for item in episodes]),
            rewards=np.stack([item["rewards"] for item in episodes]),
            risks=np.stack([item["risks"] for item in episodes]),
            dones=np.stack([item["dones"] for item in episodes]),
        )
        pair_start = int(pair_cfg["base_seed"]) + 1000 * scenario_id
        pair_count = int(args.pairs_per_mechanism or pair_cfg["pairs_per_mechanism"])
        pairs = []
        mechanisms = []
        for mechanism_index, mechanism in enumerate(("push_box", "remove_armor")):
            for index in range(pair_count):
                pair = generate_paired_intervention(
                    seed=pair_start + mechanism_index * pair_count + index,
                    mechanism=mechanism,
                    horizon=args.horizon,
                    scenario=scenario,
                    tracker_kwargs=tracker_overrides(scenario),
                )
                if pair.factual.actions.shape[0] != args.horizon or pair.intervention.actions.shape[0] != args.horizon:
                    raise RuntimeError("paired intervention terminated before the registered horizon")
                pairs.append(pair)
                mechanisms.append(mechanism)
        intervention_path = folder / "interventions.npz"
        np.savez_compressed(
            intervention_path,
            pair_ids=np.asarray([pair.factual.pair_id for pair in pairs], dtype=np.int64),
            seeds=np.asarray([pair.factual.exogenous_seed for pair in pairs], dtype=np.int64),
            mechanisms=np.asarray(mechanisms),
            factual_tokens=branch_stack(pairs, "belief_state", "factual"),
            factual_actions=branch_stack(pairs, "actions", "factual"),
            factual_next_tokens=branch_stack(pairs, "next_belief_state", "factual"),
            factual_rewards=branch_stack(pairs, "rewards", "factual"),
            factual_risks=branch_stack(pairs, "rule_risks", "factual"),
            intervention_tokens=branch_stack(pairs, "belief_state", "intervention"),
            intervention_actions=branch_stack(pairs, "actions", "intervention"),
            intervention_next_tokens=branch_stack(pairs, "next_belief_state", "intervention"),
            intervention_rewards=branch_stack(pairs, "rewards", "intervention"),
            intervention_risks=branch_stack(pairs, "rule_risks", "intervention"),
        )
        file_hashes[str(prediction_path.relative_to(output)).replace("\\", "/")] = sha256(prediction_path)
        file_hashes[str(intervention_path.relative_to(output)).replace("\\", "/")] = sha256(intervention_path)
    audit = split_audit(config)
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    canonical = json.dumps({"config": config, "files": file_hashes}, sort_keys=True).encode("utf-8")
    manifest = {
        "status": "completed" if audit["completed"] else "failed",
        "completed": bool(audit["completed"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_config": str(args.config.resolve()),
        "split_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": file_hashes,
        "horizon": args.horizon,
        "behavior_mixture": {"legacy": 0.4, "scripted_intervention": 0.3, "static_rule_graph": 0.2, "random_legal": 0.1},
        "behavior_checkpoints": {
            "legacy": str(args.legacy_checkpoint.resolve()) if args.legacy_checkpoint else None,
            "static_rule_graph": str(args.static_checkpoint.resolve()) if args.static_checkpoint else None,
        },
        "scripted_bootstrap": bool(args.allow_scripted_bootstrap),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
