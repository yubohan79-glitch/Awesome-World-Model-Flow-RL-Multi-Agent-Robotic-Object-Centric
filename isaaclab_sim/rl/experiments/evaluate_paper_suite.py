from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


RL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from evaluate_policy import actor_action, load_policy
from expert_policy import compose_policy_action
from experiments.paper_statistics import equal_mass_ece, fixed_tail_cvar
from experiments.scenario_protocol import SCENARIOS, aggressive_action, apply_scenario, tracker_overrides
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from world_model import BeliefTracker, build_typed_edges, canonical_node_types_torch, extract_rule_risks, tokens_from_flat
from world_model.belief_graph import PHYSICAL_TOKEN_DIM, TOKEN_PRESENT, TOKEN_X, TOKEN_Y
from world_model.constraint_graph_dynamics import duration_bucket_targets, edge_transition_targets, typed_edge_valid_mask


RISK_NAMES = ("collision", "penetration", "illegal_fire", "los_or_range")


def f1_scores(predicted: np.ndarray, target: np.ndarray, labels: tuple[int, ...] = (0, 1)) -> dict[str, float]:
    predicted = np.asarray(predicted).reshape(-1)
    target = np.asarray(target).reshape(-1)
    per_label = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = int(np.count_nonzero((predicted == label) & (target == label)))
        fp = int(np.count_nonzero((predicted == label) & (target != label)))
        fn = int(np.count_nonzero((predicted != label) & (target == label)))
        denominator = 2 * tp + fp + fn
        per_label.append(2 * tp / denominator if denominator else 1.0)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_denominator = 2 * total_tp + total_fp + total_fn
    return {
        "macro_f1": float(np.mean(per_label)),
        "micro_f1": float(2 * total_tp / micro_denominator if micro_denominator else 1.0),
    }


def first_change_time(edges: np.ndarray) -> np.ndarray:
    changed = edges[:, 1:] != edges[:, :-1]
    result = np.full(edges.shape[0:1] + edges.shape[2:], -1, dtype=np.int16)
    any_change = changed.any(axis=1)
    result[any_change] = changed.argmax(axis=1)[any_change] + 1
    return result


def evaluate_prediction_dataset(
    model,
    dataset_path: Path,
    *,
    device: torch.device,
    particles: int,
    batch_size: int,
) -> dict[str, object]:
    if not hasattr(model, "cbg_world_model"):
        return {"status": "not_applicable", "reason": "legacy flat-state model has no belief graph"}
    data = np.load(dataset_path, allow_pickle=False)
    tokens_np = np.asarray(data["tokens"], dtype=np.float32)
    actions_np = np.asarray(data["actions"], dtype=np.float32)
    risks_np = np.asarray(data["risks"], dtype=np.float32)
    rewards_np = np.asarray(data["rewards"], dtype=np.float32)
    model_world = model.cbg_world_model
    mean_tokens = []
    mean_edges = []
    risk_probabilities = []
    sampled_costs = []
    event_logits = []
    hazard_logits = []
    duration_logits = []
    for start in range(0, tokens_np.shape[0], batch_size):
        stop = min(start + batch_size, tokens_np.shape[0])
        initial = torch.as_tensor(tokens_np[start:stop, 0], device=device)
        actions = torch.as_tensor(actions_np[start:stop], device=device)
        with torch.no_grad():
            rollout = model_world.rollout(
                initial,
                actions,
                particles_per_member=particles,
                sample_state=True,
                return_edge_diagnostics=True,
            )
        mean_tokens.append(rollout["tokens"].mean(dim=0).cpu().numpy())
        mean_edges.append(rollout["edges"].mean(dim=0).cpu().numpy())
        risk_probabilities.append(rollout["risk_prob"].mean(dim=0).cpu().numpy())
        sampled_costs.append(rollout["risk_cost_sample"].cpu().numpy())
        event_logits.append(rollout["edge_event_logits"].mean(dim=0).cpu().numpy())
        hazard_logits.append(rollout["edge_hazard_logits"].mean(dim=0).cpu().numpy())
        duration_logits.append(rollout["edge_duration_logits"].mean(dim=0).cpu().numpy())
    prediction = np.concatenate(mean_tokens, axis=0)
    predicted_edges = np.concatenate(mean_edges, axis=0)
    risk_probability = np.concatenate(risk_probabilities, axis=0)
    cost_samples = np.concatenate(sampled_costs, axis=1)
    event_logit = np.concatenate(event_logits, axis=0)
    hazard_logit = np.concatenate(hazard_logits, axis=0)
    duration_logit = np.concatenate(duration_logits, axis=0)

    horizons: dict[str, object] = {}
    for horizon in (1, 5, 10):
        if horizon >= tokens_np.shape[1]:
            continue
        target = tokens_np[:, horizon]
        present = target[..., TOKEN_PRESENT] > 0.5
        physical_error = np.square(prediction[:, horizon, ..., :PHYSICAL_TOKEN_DIM] - target[..., :PHYSICAL_TOKEN_DIM])
        position_error = np.square(prediction[:, horizon, ..., TOKEN_X:TOKEN_Y + 1] - target[..., TOKEN_X:TOKEN_Y + 1])
        denominator = max(int(np.count_nonzero(present)), 1)
        episode_rmse = np.sqrt(
            (physical_error * present[..., None]).sum(axis=(1, 2))
            / np.maximum(present.sum(axis=1) * PHYSICAL_TOKEN_DIM, 1)
        )
        horizons[str(horizon)] = {
            "physical_rmse": float(np.sqrt((physical_error * present[..., None]).sum() / (denominator * PHYSICAL_TOKEN_DIM))),
            "position_rmse_normalized": float(np.sqrt((position_error * present[..., None]).sum() / (denominator * 2))),
            "episode_physical_rmse_mean": float(episode_rmse.mean()),
            "episode_physical_rmse_std": float(episode_rmse.std(ddof=1)),
        }

    token_tensor = torch.as_tensor(tokens_np)
    batch, time_steps = token_tensor.shape[:2]
    types = canonical_node_types_torch(batch, "cpu")
    target_edges = torch.stack(
        [build_typed_edges(token_tensor[:, step], types) for step in range(time_steps)], dim=1
    )
    target_edge_np = target_edges.numpy().astype(np.int8)
    valid = typed_edge_valid_mask(types).numpy()
    valid_time = np.broadcast_to(valid[:, None], target_edge_np.shape)
    predicted_binary = predicted_edges >= 0.5
    presence = f1_scores(predicted_binary[valid_time], target_edge_np[valid_time], (0, 1))
    target_events = edge_transition_targets(target_edges[:, :-1], target_edges[:, 1:]).numpy()
    predicted_events = event_logit.argmax(axis=-1)
    event_valid = np.broadcast_to(valid[:, None], target_events.shape)
    event_scores = f1_scores(predicted_events[event_valid], target_events[event_valid], (0, 1, 2))
    duration_target = duration_bucket_targets(target_edges)[:, :-1].numpy()
    duration_predicted = duration_logit.argmax(axis=-1)
    duration_valid = event_valid & (duration_target >= 0)
    duration_scores = (
        f1_scores(duration_predicted[duration_valid], duration_target[duration_valid], (0, 1, 2, 3))
        if duration_valid.any()
        else {"macro_f1": None, "micro_f1": None}
    )
    hazard_probability = 1.0 / (1.0 + np.exp(-np.clip(hazard_logit, -30.0, 30.0)))
    hazard_target = ((target_edge_np[:, :-1] > 0) & (target_edge_np[:, 1:] == 0)).astype(np.float32)
    hazard_valid = event_valid & (target_edge_np[:, :-1] > 0)
    survival_brier = float(np.square(hazard_probability[hazard_valid] - hazard_target[hazard_valid]).mean()) if hazard_valid.any() else None
    predicted_change = first_change_time(predicted_binary.astype(np.int8))
    actual_change = first_change_time(target_edge_np)
    change_valid = (predicted_change >= 0) | (actual_change >= 0)
    event_time_mae = float(np.abs(predicted_change[change_valid] - actual_change[change_valid]).mean()) if change_valid.any() else 0.0

    calibration = {}
    for risk_index, name in enumerate(RISK_NAMES):
        probability = risk_probability[..., risk_index].reshape(-1)
        target = risks_np[..., risk_index].reshape(-1)
        positives = int(np.count_nonzero(target > 0.5))
        calibration[name] = {
            "brier": float(np.square(probability - target).mean()),
            "binary_nll": float(np.mean(-(target * np.log(np.clip(probability, 1e-8, 1.0)) + (1.0 - target) * np.log(np.clip(1.0 - probability, 1e-8, 1.0))))),
            "ece_15_equal_mass": equal_mass_ece(probability, target, bins=15),
            "positive_transitions": positives,
            "low_support": positives < 20,
        }
    discounts = np.power(0.995, np.arange(actions_np.shape[1], dtype=np.float64))
    predicted_episode_cost = (cost_samples * discounts[None, None, :, None, None]).sum(axis=(2, 3))
    realized_episode_cost = (risks_np * discounts[None, :, None, None]).sum(axis=(1, 2))
    cvar = {}
    for risk_index, name in enumerate(RISK_NAMES):
        predicted_values = predicted_episode_cost[..., risk_index].reshape(-1)
        realized_values = realized_episode_cost[..., risk_index]
        predicted_cvar = fixed_tail_cvar(predicted_values, 0.90)
        realized_cvar = fixed_tail_cvar(realized_values, 0.90)
        cvar[name] = {
            "predicted": predicted_cvar,
            "realized": realized_cvar,
            "absolute_error": abs(predicted_cvar - realized_cvar),
        }
    return {
        "status": "completed",
        "episodes": int(tokens_np.shape[0]),
        "horizons": horizons,
        "edge_presence": presence,
        "edge_events": event_scores,
        "duration_bucket": duration_scores,
        "event_time_mae_steps": event_time_mae,
        "survival_integrated_brier": survival_brier,
        "risk_calibration": calibration,
        "cvar_0_90": cvar,
        "reward_rmse": float(np.sqrt(np.square(rewards_np - rewards_np.mean(axis=0, keepdims=True)).mean())),
    }


def evaluate_interventions(
    model,
    dataset_path: Path,
    *,
    device: torch.device,
    particles: int,
    batch_size: int,
) -> dict[str, object]:
    if not hasattr(model, "cbg_world_model"):
        return {"status": "not_applicable", "reason": "legacy flat-state model has no intervention head"}
    data = np.load(dataset_path, allow_pickle=False)
    mechanisms = np.asarray(data["mechanisms"])
    metrics: dict[str, object] = {}
    for mechanism in ("push_box", "remove_armor"):
        indices = np.flatnonzero(mechanisms == mechanism)
        predicted_edge_effect = []
        actual_edge_effect = []
        predicted_return_effect = []
        actual_return_effect = []
        predicted_risk_effect = []
        actual_risk_effect = []
        for start in range(0, indices.size, batch_size):
            selection = indices[start:start + batch_size]
            branch_outputs = []
            for prefix in ("factual", "intervention"):
                initial = tokens_from_flat(
                    torch.as_tensor(data[f"{prefix}_tokens"][selection, 0], device=device)
                )
                actions = torch.as_tensor(data[f"{prefix}_actions"][selection], device=device)
                with torch.no_grad():
                    branch_outputs.append(
                        model.cbg_world_model.rollout(
                            initial, actions, particles_per_member=particles, sample_state=True
                        )
                    )
            factual, intervention = branch_outputs
            predicted_edge_effect.append(
                (intervention["edges"][:, :, -1].mean(dim=0) - factual["edges"][:, :, -1].mean(dim=0)).sum(dim=(1, 2, 3)).cpu().numpy()
            )
            predicted_return_effect.append(
                (intervention["rewards"].sum(dim=(2, 3)).mean(dim=0) - factual["rewards"].sum(dim=(2, 3)).mean(dim=0)).cpu().numpy()
            )
            predicted_risk_effect.append(
                (intervention["risk_prob"].sum(dim=(2, 3, 4)).mean(dim=0) - factual["risk_prob"].sum(dim=(2, 3, 4)).mean(dim=0)).cpu().numpy()
            )
            node_types = canonical_node_types_torch(selection.size, "cpu")
            factual_final = tokens_from_flat(torch.as_tensor(data["factual_next_tokens"][selection, -1]))
            intervention_final = tokens_from_flat(torch.as_tensor(data["intervention_next_tokens"][selection, -1]))
            actual_edge_effect.append(
                (build_typed_edges(intervention_final, node_types) - build_typed_edges(factual_final, node_types)).sum(dim=(1, 2, 3)).numpy()
            )
            actual_return_effect.append(
                (data["intervention_rewards"][selection].sum(axis=(1, 2)) - data["factual_rewards"][selection].sum(axis=(1, 2)))
            )
            actual_risk_effect.append(
                (data["intervention_risks"][selection].sum(axis=(1, 2, 3)) - data["factual_risks"][selection].sum(axis=(1, 2, 3)))
            )
        predicted_edge = np.concatenate(predicted_edge_effect)
        actual_edge = np.concatenate(actual_edge_effect)
        predicted_return = np.concatenate(predicted_return_effect)
        actual_return = np.concatenate(actual_return_effect)
        predicted_risk = np.concatenate(predicted_risk_effect)
        actual_risk = np.concatenate(actual_risk_effect)
        predicted_choice = predicted_return - 2.0 * predicted_risk > 0.0
        actual_choice = actual_return - 2.0 * actual_risk > 0.0
        actual_utility = actual_return - 2.0 * actual_risk
        metrics[mechanism] = {
            "pairs": int(indices.size),
            "edge_effect_sign_accuracy": float(np.mean(np.sign(predicted_edge) == np.sign(actual_edge))),
            "edge_effect_magnitude_mae": float(np.mean(np.abs(predicted_edge - actual_edge))),
            "return_effect_sign_accuracy": float(np.mean(np.sign(predicted_return) == np.sign(actual_return))),
            "risk_effect_sign_accuracy": float(np.mean(np.sign(predicted_risk) == np.sign(actual_risk))),
            "action_selection_accuracy": float(np.mean(predicted_choice == actual_choice)),
            "intervention_regret": float(np.mean(np.where(predicted_choice == actual_choice, 0.0, np.abs(actual_utility)))),
        }
    return {"status": "completed", "mechanisms": metrics, "claim_scope": "controlled intervention consistency, not causal identification"}


def raw_actor(model, observations, team: str, device: torch.device) -> np.ndarray:
    return actor_action(model, observations[team], team, device, True)


def run_match(
    candidate,
    opponent,
    *,
    opponent_name: str,
    ego_team: str,
    seed: int,
    scenario: str,
    max_steps: int,
    device: torch.device,
) -> dict[str, object]:
    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=False, action_shield=True)
    observations, _ = env.reset(seed=seed)
    apply_scenario(env, scenario, seed)
    observations = {team: env._obs(team) for team in AGENTS}
    opponent_team = "blue" if ego_team == "yellow" else "yellow"
    config = getattr(candidate, "cbg_belief_config", {})
    tracker_args = dict(config)
    tracker_args.update(tracker_overrides(scenario))
    tracker = BeliefTracker(seed=seed, **tracker_args)
    planner = getattr(candidate, "cbg_planner", None)
    cumulative_cost = np.zeros(len(RISK_NAMES), dtype=np.float64)
    cumulative_reward = 0.0
    shield_interventions = 0
    planner_cvar = []
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        if planner is None:
            ego_raw = raw_actor(candidate, observations, ego_team, device)
        else:
            joint_obs = torch.as_tensor(
                np.stack([observations[team] for team in AGENTS])[None], dtype=torch.float32, device=device
            )
            belief = torch.as_tensor(tracker.observe(env).tokens[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                plan = planner.plan(candidate, joint_obs, belief)
            ego_index = AGENTS.index(ego_team)
            ego_raw = plan.actions[0, ego_index].cpu().numpy().astype(np.float32)
            selected = int(plan.candidate_indices[0, ego_index].item())
            planner_cvar.append(plan.cvar_cost[0, ego_index, selected].cpu().numpy())
        if opponent_name == "aggressive_scripted":
            opponent_raw = aggressive_action()
        else:
            opponent_raw = raw_actor(opponent, observations, opponent_team, device)
        raw = {ego_team: ego_raw, opponent_team: opponent_raw}
        executed = {
            team: compose_policy_action(
                env,
                team,
                raw[team],
                policy_mode="residual_expert",
                residual_scale=0.04,
            )
            for team in AGENTS
        }
        observations, rewards, terminations, truncations, infos = env.step(executed)
        risks = extract_rule_risks(infos, executed)
        ego_index = AGENTS.index(ego_team)
        cumulative_cost += (0.995 ** (step - 1)) * risks[ego_index]
        cumulative_reward += (0.995 ** (step - 1)) * float(rewards[ego_team])
        shield_interventions += sum(
            int(bool(infos[team].get("action_shield_contact") or infos[team].get("action_shield_fire")))
            for team in AGENTS
        )
        if any(terminations.values()) or any(truncations.values()):
            break
    return {
        "world_seed": seed,
        "scenario": scenario,
        "opponent": opponent_name,
        "ego_team": ego_team,
        "winner": env.winner or "draw",
        "steps": step,
        "discounted_return": cumulative_reward,
        **{f"cost_{name}": float(cumulative_cost[index]) for index, name in enumerate(RISK_NAMES)},
        "shield_interventions": shield_interventions,
        "shield_intervention_rate": shield_interventions / max(step * len(AGENTS), 1),
        "predicted_cvar_total": float(np.mean(np.sum(planner_cvar, axis=1))) if planner_cvar else None,
        "wall_time_s": time.perf_counter() - started,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one registered checkpoint-scenario cell.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--frozen-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-opponent-one", type=Path)
    parser.add_argument("--legacy-opponent-two", type=Path)
    parser.add_argument("--aggressive-opponent-checkpoint", type=Path)
    parser.add_argument("--world-seeds", type=int, default=32)
    parser.add_argument("--matches", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--allow-scripted-opponents", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    candidate, checkpoint = load_policy(args.checkpoint, device)
    opponent_paths = (
        args.legacy_opponent_one,
        args.legacy_opponent_two,
        args.aggressive_opponent_checkpoint,
    )
    protocol_valid = all(path is not None and path.is_file() for path in opponent_paths)
    if not protocol_valid and not args.allow_scripted_opponents:
        raise ValueError("formal evaluation requires two frozen legacy and one aggression-shaped opponent checkpoint")
    loaded = [load_policy(path, device)[0] if path is not None and path.is_file() else None for path in opponent_paths]
    opponents = (
        ("legacy_seed_1", loaded[0]),
        ("legacy_seed_2", loaded[1]),
        ("aggressive_scripted", None),
        ("aggression_shaped_sac", loaded[2]),
    )
    frozen_folder = args.frozen_data_root / args.scenario
    prediction = evaluate_prediction_dataset(
        candidate,
        frozen_folder / "prediction.npz",
        device=device,
        particles=args.particles,
        batch_size=args.prediction_batch_size,
    )
    interventions = evaluate_interventions(
        candidate,
        frozen_folder / "interventions.npz",
        device=device,
        particles=args.particles,
        batch_size=args.prediction_batch_size,
    )
    scenario_id = SCENARIOS.index(args.scenario)
    seed_start = 430000 + 1000 * scenario_id
    rows = []
    for world_seed in range(seed_start, seed_start + args.world_seeds):
        for opponent_name, opponent in opponents:
            for ego_team in AGENTS:
                effective_opponent = opponent
                effective_name = opponent_name
                if effective_opponent is None and opponent_name != "aggressive_scripted":
                    effective_name = "aggressive_scripted"
                rows.append(
                    run_match(
                        candidate,
                        effective_opponent,
                        opponent_name=effective_name,
                        ego_team=ego_team,
                        seed=world_seed,
                        scenario=args.scenario,
                        max_steps=args.max_steps,
                        device=device,
                    )
                )
    if len(rows) != args.matches:
        raise RuntimeError(f"registered matrix produced {len(rows)} matches, expected {args.matches}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "episodes.csv", rows)
    win_scores = [
        1.0 if row["winner"] == row["ego_team"] else 0.5 if row["winner"] == "draw" else 0.0
        for row in rows
    ]
    realized_cvar = {
        name: fixed_tail_cvar([float(row[f"cost_{name}"]) for row in rows], 0.90)
        for name in RISK_NAMES
    }
    completed = len(rows) == 256 and protocol_valid
    summary = {
        "status": "completed" if completed else "smoke_completed",
        "completed": completed,
        "variant": args.variant,
        "training_seed": args.seed,
        "scenario": args.scenario,
        "checkpoint_training_variant": checkpoint.get("training_variant", checkpoint.get("config", {}).get("training_variant")),
        "match_count": len(rows),
        "opponent_protocol_valid": protocol_valid,
        "planner_mode": "ego_only_cvar_mpc" if hasattr(candidate, "cbg_planner") else "actor_only",
        "win_score": float(np.mean(win_scores)),
        "win_rate": float(np.mean([row["winner"] == row["ego_team"] for row in rows])),
        "draw_rate": float(np.mean([row["winner"] == "draw" for row in rows])),
        "realized_cvar_0_90": realized_cvar,
        "prediction": prediction,
        "interventions": interventions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "variant", "scenario", "match_count", "win_score")}, indent=2))
    return 0 if completed or args.allow_scripted_opponents else 1


if __name__ == "__main__":
    raise SystemExit(main())
