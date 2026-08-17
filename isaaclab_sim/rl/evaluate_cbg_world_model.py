from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_policy import load_policy
from experiments.scenario_protocol import SCENARIOS, apply_scenario, tracker_overrides
from expert_policy import compose_policy_action
from robocup_visionrl_selfplay_env import (
    AGENTS,
    DomainRandomizationParams,
    RoboCupVisionRLSelfPlayEnv,
)
from train_world_model_sacflow_selfplay import MultiAgentFlowActors
from world_model import BeliefTracker, EdgeType, NodeType, extract_rule_risks
from world_model.belief_graph import (
    ARMOR_BLOCKER_SLICE,
    BOX_SLICE,
    PHYSICAL_TOKEN_DIM,
    TOKEN_EXTENT_X,
    TOKEN_EXTENT_Y,
    TOKEN_PRESENT,
    TOKEN_X,
    TOKEN_Y,
    build_typed_edges,
    canonical_node_types_torch,
)

def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probabilities = probabilities.detach().reshape(-1).clamp(0.0, 1.0)
    labels = labels.detach().reshape(-1)
    error = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= boundaries[index]) & (probabilities <= boundaries[index + 1])
        else:
            mask = (probabilities >= boundaries[index]) & (probabilities < boundaries[index + 1])
        if mask.any():
            error = error + mask.float().mean() * (probabilities[mask].mean() - labels[mask].mean()).abs()
    return float(error.cpu())


def binary_auroc(probabilities: torch.Tensor, labels: torch.Tensor) -> float | None:
    probabilities = probabilities.detach().reshape(-1)
    labels = labels.detach().reshape(-1) > 0.5
    positive = probabilities[labels]
    negative = probabilities[~labels]
    if positive.numel() == 0 or negative.numel() == 0:
        return None
    comparisons = (positive[:, None] > negative[None, :]).float()
    ties = (positive[:, None] == negative[None, :]).float()
    return float((comparisons + 0.5 * ties).mean().cpu())


def upper_tail_mean(values: torch.Tensor, alpha: float, dim: int = 0) -> torch.Tensor:
    count = max(1, int(np.ceil(values.shape[dim] * float(np.clip(alpha, 1e-6, 1.0)))))
    sorted_values = torch.sort(values, dim=dim, descending=True).values
    return sorted_values.narrow(dim, 0, count).mean(dim=dim)


def apply_ood_scenario(env: RoboCupVisionRLSelfPlayEnv, scenario: str) -> None:
    apply_scenario(env, scenario, 0)


def environment_counterfactual_geometry(seed: int, scenario: str) -> dict[str, object]:
    """Evaluate the intervention directions against the simulator geometry."""

    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=False, action_shield=True)
    env.reset(seed=seed)
    apply_scenario(env, scenario, seed)
    box_names = sorted(env.pushable_obstacles)
    box_result: dict[str, object] = {"pair_found": False}
    if box_names:
        original_boxes = {name: value.copy() for name, value in env.pushable_obstacles.items()}
        corners = (np.array([-0.92, 0.92], dtype=np.float32), np.array([0.92, -0.92], dtype=np.float32))
        for index, name in enumerate(box_names):
            env.pushable_obstacles[name] = corners[index % len(corners)].copy()
        for team in AGENTS:
            origin = (float(env.poses[team][0]), float(env.poses[team][1]))
            opponent = "blue" if team == "yellow" else "yellow"
            for target in env.targets:
                if target.owner != opponent or target.knocked:
                    continue
                if env._line_blocked(origin, target.xy):
                    continue
                midpoint = 0.5 * (env.poses[team][:2] + np.asarray(target.xy, dtype=np.float32))
                env.pushable_obstacles[box_names[0]] = midpoint.astype(np.float32)
                blocked_before = env._line_blocked(origin, target.xy)
                env.pushable_obstacles[box_names[0]] = corners[0].copy()
                clear_after = not env._line_blocked(origin, target.xy)
                if blocked_before and clear_after:
                    box_result = {
                        "pair_found": True,
                        "team": team,
                        "target": target.name,
                        "line_blocked_before_push": True,
                        "line_clear_after_push": True,
                    }
                    break
            if box_result["pair_found"]:
                break
        env.pushable_obstacles = original_boxes

    armor_result: dict[str, object] = {"pair_found": False}
    original_armor = dict(env.armor)
    original_boxes = {name: value.copy() for name, value in env.pushable_obstacles.items()}
    for index, name in enumerate(sorted(env.pushable_obstacles)):
        env.pushable_obstacles[name] = (
            np.array([-0.96, 0.96], dtype=np.float32)
            if index % 2 == 0
            else np.array([0.96, -0.96], dtype=np.float32)
        )
    for team in AGENTS:
        opponent = "blue" if team == "yellow" else "yellow"
        base = next(target for target in env.targets if target.kind == f"base_{opponent}")
        env.armor[opponent] = 4
        candidates = env._candidate_base_fire_poses(team, base, risk=0.75)
        for candidate in candidates:
            origin = env._laser_origin_for_fire_pose(candidate, np.asarray(base.xy, dtype=np.float32))
            if not env._line_blocked(origin, base.xy):
                continue
            clear_at = None
            for armor_count in (3, 2, 1, 0):
                env.armor[opponent] = armor_count
                if not env._line_blocked(origin, base.xy):
                    clear_at = armor_count
                    break
            env.armor[opponent] = 4
            if clear_at is not None:
                armor_result = {
                    "pair_found": True,
                    "attacker": team,
                    "base": base.name,
                    "line_blocked_with_full_armor": True,
                    "line_clear_after_armor_hits": True,
                    "armor_remaining_when_clear": clear_at,
                    "armor_hits_required": 4 - clear_at,
                }
                break
        if armor_result["pair_found"]:
            break
    env.armor = original_armor
    env.pushable_obstacles = original_boxes
    return {"push_box": box_result, "remove_armor": armor_result}


def _tracker_for_scenario(seed: int, scenario: str, config: dict[str, object]) -> BeliefTracker:
    kwargs = {
        "max_age_s": float(config.get("belief_max_age_s", 3.0)),
        "covariance_growth": float(config.get("belief_covariance_growth", 0.08)),
        "sensor_delay_steps": int(config.get("sensor_delay_steps", 1)),
        "observation_dropout": float(config.get("observation_dropout", 0.05)),
    }
    kwargs.update(tracker_overrides(scenario))
    return BeliefTracker(seed=seed, **kwargs)


@torch.no_grad()
def collect_episode(
    actors: MultiAgentFlowActors,
    *,
    seed: int,
    scenario: str,
    max_steps: int,
    device: torch.device,
    policy_mode: str,
    residual_scale: float,
    belief_config: dict[str, object],
) -> dict[str, object]:
    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=False, action_shield=True)
    observations, _info = env.reset(seed=seed)
    apply_scenario(env, scenario, seed)
    tracker = _tracker_for_scenario(seed, scenario, belief_config)
    token_sequence = [tracker.observe(env).tokens]
    action_sequence = []
    risk_sequence = []
    reward_sequence = []

    for _step in range(max_steps):
        obs_tensor = torch.as_tensor(
            np.stack([observations[team] for team in AGENTS])[None], dtype=torch.float32, device=device
        )
        raw_tensor = actors.deterministic(obs_tensor)
        raw_actions = raw_tensor[0].cpu().numpy().astype(np.float32)
        if scenario == "aggressive_opponent":
            raw_actions[1] = np.asarray([0.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        executed = {
            team: compose_policy_action(
                env,
                team,
                raw_actions[index],
                policy_mode=policy_mode,
                residual_scale=residual_scale,
            )
            for index, team in enumerate(AGENTS)
        }
        observations, rewards, terminations, truncations, infos = env.step(executed)
        action_sequence.append(raw_actions)
        risk_sequence.append(extract_rule_risks(infos, executed))
        reward_sequence.append(np.asarray([rewards[team] for team in AGENTS], dtype=np.float32))
        token_sequence.append(tracker.observe(env).tokens)
        if any(terminations.values()) or any(truncations.values()):
            break

    return {
        "tokens": np.asarray(token_sequence, dtype=np.float32),
        "actions": np.asarray(action_sequence, dtype=np.float32),
        "risks": np.asarray(risk_sequence, dtype=np.float32),
        "rewards": np.asarray(reward_sequence, dtype=np.float32),
        "winner": env.winner or "timeout",
        "scores": dict(env.scores),
    }


def make_windows(episodes: list[dict[str, object]], horizon: int) -> tuple[np.ndarray, ...]:
    token_windows = []
    action_windows = []
    risk_windows = []
    reward_windows = []
    stride = max(horizon // 2, 1)
    for episode in episodes:
        tokens = np.asarray(episode["tokens"])
        actions = np.asarray(episode["actions"])
        risks = np.asarray(episode["risks"])
        rewards = np.asarray(episode["rewards"])
        for start in range(0, max(0, len(actions) - horizon + 1), stride):
            token_windows.append(tokens[start : start + horizon + 1])
            action_windows.append(actions[start : start + horizon])
            risk_windows.append(risks[start : start + horizon])
            reward_windows.append(rewards[start : start + horizon])
    if not token_windows:
        raise RuntimeError(f"no complete windows of horizon {horizon}; increase --max-steps")
    return (
        np.asarray(token_windows, dtype=np.float32),
        np.asarray(action_windows, dtype=np.float32),
        np.asarray(risk_windows, dtype=np.float32),
        np.asarray(reward_windows, dtype=np.float32),
    )


@torch.no_grad()
def evaluate_windows(
    world_model,
    token_windows: np.ndarray,
    action_windows: np.ndarray,
    risk_windows: np.ndarray,
    reward_windows: np.ndarray,
    device: torch.device,
    horizons: tuple[int, ...],
    cvar_tail_fraction: float,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    tokens = torch.as_tensor(token_windows, device=device)
    actions = torch.as_tensor(action_windows, device=device)
    risk_targets = torch.as_tensor(risk_windows, device=device)
    reward_targets = torch.as_tensor(reward_windows, device=device)
    rollout = world_model.rollout(tokens[:, 0], actions)
    predicted_tokens = rollout["tokens"].mean(dim=0)
    predicted_rewards = rollout["rewards"].mean(dim=0)
    predicted_risks = rollout["risk_prob"].mean(dim=0)

    prediction_metrics: dict[str, object] = {}
    for horizon in horizons:
        if horizon >= tokens.shape[1]:
            continue
        target = tokens[:, horizon]
        prediction = predicted_tokens[:, horizon]
        present = target[..., TOKEN_PRESENT].unsqueeze(-1).clamp(0.0, 1.0)
        physical_error = (prediction[..., :PHYSICAL_TOKEN_DIM] - target[..., :PHYSICAL_TOKEN_DIM]).square()
        xy_error = (prediction[..., TOKEN_X:TOKEN_Y + 1] - target[..., TOKEN_X:TOKEN_Y + 1]).square()
        denominator = present.sum().clamp_min(1.0)
        prediction_metrics[str(horizon)] = {
            "physical_rmse": float(torch.sqrt((physical_error * present).sum() / (denominator * PHYSICAL_TOKEN_DIM)).cpu()),
            "position_rmse": float(torch.sqrt((xy_error * present).sum() / (denominator * 2.0)).cpu()),
            "epistemic_variance": float(
                rollout["tokens"][:, :, horizon, ..., :PHYSICAL_TOKEN_DIM].var(dim=0, unbiased=False).mean().cpu()
            ),
            "aleatoric_variance": float(rollout["aleatoric_var"][:, :, horizon - 1].mean().cpu()),
        }

    brier = (predicted_risks - risk_targets).square().mean(dim=(0, 1))
    risk_names = ("collision", "penetration", "illegal_fire", "los_or_range")
    calibration = {}
    for index, name in enumerate(risk_names):
        probabilities = predicted_risks[..., index]
        labels = risk_targets[..., index]
        calibration[name] = {
            "brier": float(brier[:, index].mean().cpu()),
            "ece": expected_calibration_error(probabilities, labels),
            "auroc": binary_auroc(probabilities, labels),
        }

    member_rule_risk = rollout["risk_prob"].mean(dim=(2, 3))
    upper_cvar = upper_tail_mean(member_rule_risk, cvar_tail_fraction, dim=0)
    metrics = {
        "windows": int(tokens.shape[0]),
        "prediction": prediction_metrics,
        "reward_rmse": float(torch.sqrt((predicted_rewards - reward_targets).square().mean()).cpu()),
        "risk_calibration": calibration,
        "cvar_rule_risk": {
            name: float(upper_cvar[:, index].mean().cpu()) for index, name in enumerate(risk_names)
        },
    }
    return metrics, {"initial_tokens": tokens[:, 0], "actions": actions, "rollout": rollout}


@torch.no_grad()
def counterfactual_probes(
    world_model,
    sample: dict[str, torch.Tensor],
    environment_geometry: dict[str, object],
) -> dict[str, object]:
    all_initial = sample["initial_tokens"]
    all_types = canonical_node_types_torch(all_initial.shape[0], all_initial.device)
    all_edges = build_typed_edges(all_initial, all_types)
    interaction_count = (
        all_edges[:, EdgeType.BLOCKS_ROUTE].sum(dim=(1, 2))
        + all_edges[:, EdgeType.PROTECTS_BASE].sum(dim=(1, 2))
    )
    sample_index = int(interaction_count.argmax().item())
    initial = all_initial[sample_index : sample_index + 1]
    actions = sample["actions"][sample_index : sample_index + 1]
    node_types = canonical_node_types_torch(1, initial.device)
    factual_edges = build_typed_edges(initial, node_types)
    constructed_box_baseline = False
    box_candidates = torch.where(initial[0, BOX_SLICE, TOKEN_PRESENT] > 0.5)[0]
    if factual_edges[:, EdgeType.BLOCKS_ROUTE].sum() == 0 and box_candidates.numel() > 0:
        robots = torch.where(node_types[0] == int(NodeType.ROBOT))[0]
        targets = torch.where(
            (node_types[0] == int(NodeType.TARGET)) & (initial[0, :, TOKEN_PRESENT] > 0.5)
        )[0]
        if robots.numel() > 0 and targets.numel() > 0:
            box_index = BOX_SLICE.start + int(box_candidates[0])
            initial = initial.clone()
            initial[0, box_index, TOKEN_X:TOKEN_Y + 1] = 0.5 * (
                initial[0, robots[0], TOKEN_X:TOKEN_Y + 1]
                + initial[0, targets[0], TOKEN_X:TOKEN_Y + 1]
            )
            initial[0, box_index, TOKEN_EXTENT_X:TOKEN_EXTENT_Y + 1] = (
                initial[0, box_index, TOKEN_EXTENT_X:TOKEN_EXTENT_Y + 1].clamp_min(0.08)
            )
            factual_edges = build_typed_edges(initial, node_types)
            constructed_box_baseline = True
    factual = world_model.rollout(initial, actions)

    moved_box = initial.clone()
    moved_edges = factual_edges
    box_candidates = torch.where(initial[0, BOX_SLICE, TOKEN_PRESENT] > 0.5)[0]
    best_box_score = float(factual_edges[:, EdgeType.BLOCKS_ROUTE].sum().item())
    for local_index in box_candidates.tolist():
        box_index = BOX_SLICE.start + int(local_index)
        for x_value in (-0.98, 0.98):
            for y_value in (-0.98, 0.98):
                candidate = initial.clone()
                candidate[0, box_index, TOKEN_X] = x_value
                candidate[0, box_index, TOKEN_Y] = y_value
                candidate_edges = build_typed_edges(candidate, node_types)
                score = float(candidate_edges[:, EdgeType.BLOCKS_ROUTE].sum().item())
                if score < best_box_score:
                    moved_box, moved_edges, best_box_score = candidate, candidate_edges, score
        for feature in (TOKEN_X, TOKEN_Y):
            for offset in (-0.35, 0.35):
                candidate = initial.clone()
                candidate[0, box_index, feature] = (candidate[0, box_index, feature] + offset).clamp(-1.0, 1.0)
                candidate_edges = build_typed_edges(candidate, node_types)
                score = float(candidate_edges[:, EdgeType.BLOCKS_ROUTE].sum().item())
                if score < best_box_score:
                    moved_box, moved_edges, best_box_score = candidate, candidate_edges, score
    pushed = world_model.rollout(moved_box, actions)

    removed_armor = initial.clone()
    armor_edges = factual_edges
    armor_candidates = torch.where(removed_armor[0, ARMOR_BLOCKER_SLICE, TOKEN_PRESENT] > 0.5)[0]
    factual_los = float(factual_edges[:, EdgeType.LINE_OF_SIGHT].sum().item())
    factual_protection = float(factual_edges[:, EdgeType.PROTECTS_BASE].sum().item())
    best_armor_score = -float("inf")
    for local_index in armor_candidates.tolist():
        armor_index = ARMOR_BLOCKER_SLICE.start + int(local_index)
        candidate = initial.clone()
        candidate[0, armor_index, TOKEN_PRESENT] = 0.0
        candidate_edges = build_typed_edges(candidate, node_types)
        los_gain = float(candidate_edges[:, EdgeType.LINE_OF_SIGHT].sum().item()) - factual_los
        protection_removed = factual_protection - float(
            candidate_edges[:, EdgeType.PROTECTS_BASE].sum().item()
        )
        score = los_gain + protection_removed
        if score > best_armor_score:
            removed_armor, armor_edges, best_armor_score = candidate, candidate_edges, score
    armor_hit = world_model.rollout(removed_armor, actions)

    def outcome_delta(intervention: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            "predicted_return_delta": float(
                (intervention["rewards"].sum(dim=(2, 3)).mean() - factual["rewards"].sum(dim=(2, 3)).mean()).cpu()
            ),
            "predicted_rule_risk_delta": float(
                (intervention["risk_prob"].mean() - factual["risk_prob"].mean()).cpu()
            ),
        }

    push_direction = int(factual_edges[:, EdgeType.BLOCKS_ROUTE].sum().item()) > int(
        moved_edges[:, EdgeType.BLOCKS_ROUTE].sum().item()
    )
    armor_direction = int(armor_edges[:, EdgeType.LINE_OF_SIGHT].sum().item()) > int(
        factual_edges[:, EdgeType.LINE_OF_SIGHT].sum().item()
    )
    push_actual = environment_geometry["push_box"]
    armor_actual = environment_geometry["remove_armor"]
    return {
        "push_vs_no_push": {
            "constructed_blocking_baseline": constructed_box_baseline,
            "blocks_route_edges_before": int(factual_edges[:, EdgeType.BLOCKS_ROUTE].sum().item()),
            "blocks_route_edges_after": int(moved_edges[:, EdgeType.BLOCKS_ROUTE].sum().item()),
            "model_direction": "less_blocking_after_push" if push_direction else "no_less_blocking",
            "environment_geometry": push_actual,
            "direction_agrees_with_environment": bool(push_direction and push_actual.get("pair_found", False)),
            **outcome_delta(pushed),
        },
        "armor_removed_vs_present": {
            "protects_base_edges_before": int(factual_edges[:, EdgeType.PROTECTS_BASE].sum().item()),
            "protects_base_edges_after": int(armor_edges[:, EdgeType.PROTECTS_BASE].sum().item()),
            "los_edges_before": int(factual_edges[:, EdgeType.LINE_OF_SIGHT].sum().item()),
            "los_edges_after": int(armor_edges[:, EdgeType.LINE_OF_SIGHT].sum().item()),
            "model_direction": "more_los_after_armor_removal" if armor_direction else "no_los_gain",
            "environment_geometry": armor_actual,
            "direction_agrees_with_environment": bool(armor_direction and armor_actual.get("pair_found", False)),
            **outcome_delta(armor_hit),
        },
        "interpretation": "State interventions test model sensitivity; they are not proof of causal identification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CBG-WM prediction, calibration, OOD and interventions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("../output/eval/cbg_world_model_metrics.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    actors, checkpoint = load_policy(args.checkpoint, device)
    if str(checkpoint.get("algorithm")) != "cbg_wm_sac_flow_selfplay":
        raise ValueError("world-model evaluation requires a cbg_wm_sac_flow_selfplay checkpoint")
    world_model = actors.cbg_world_model
    config = checkpoint.get("config", {})
    episodes = [
        collect_episode(
            actors,
            seed=args.seed + index,
            scenario=args.scenario,
            max_steps=args.max_steps,
            device=device,
            policy_mode=str(config.get("policy_mode", "residual_expert")),
            residual_scale=float(config.get("residual_scale", 0.04)),
            belief_config=config,
        )
        for index in range(args.episodes)
    ]
    windows = make_windows(episodes, args.horizon)
    horizons = tuple(value for value in (1, 5, 10) if value <= args.horizon)
    metrics, sample = evaluate_windows(
        world_model,
        *windows,
        device,
        horizons,
        1.0 - float(config.get("cvar_beta", 0.90)),
    )
    result = {
        "algorithm": checkpoint["algorithm"],
        "scenario": args.scenario,
        "episodes": args.episodes,
        "task_outcomes": {
            "yellow_win_rate": sum(episode["winner"] == "yellow" for episode in episodes) / len(episodes),
            "blue_win_rate": sum(episode["winner"] == "blue" for episode in episodes) / len(episodes),
            "mean_score": {
                team: float(np.mean([episode["scores"][team] for episode in episodes])) for team in AGENTS
            },
        },
        **metrics,
        "counterfactual_probes": counterfactual_probes(
            world_model,
            sample,
            environment_counterfactual_geometry(args.seed, args.scenario),
        ),
    }
    output = args.output if args.output.is_absolute() else (Path(__file__).resolve().parent / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
