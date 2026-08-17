from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from expert_policy import compose_policy_action
from planning import FlowProposalRiskMPC
from policies import FlowActor
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from train_world_model_sacflow_selfplay import MultiAgentFlowActors
from world_model import (
    BeliefTracker,
    CounterfactualBeliefGraphWorldModel,
)


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    algorithm = str(checkpoint.get("algorithm", ""))
    if algorithm in ("object_centric_world_model_sac_flow_selfplay", "cbg_wm_sac_flow_selfplay"):
        config = checkpoint.get("config", {})
        actor_mode = str(checkpoint.get("actor_mode", config.get("actor_mode", "dual")))
        model = MultiAgentFlowActors(
            int(checkpoint["obs_dim"]),
            int(checkpoint["action_dim"]),
            int(config.get("hidden_dim", 256)),
            actor_mode=actor_mode,
            flow_steps=int(config.get("flow_steps", 3)),
            velocity_scale=float(config.get("flow_velocity_scale", 0.20)),
        ).to(device)
        model.load_state_dict(checkpoint["actor_state_dict"])
        if algorithm == "cbg_wm_sac_flow_selfplay":
            world_model = CounterfactualBeliefGraphWorldModel(
                int(checkpoint["action_dim"]),
                len(checkpoint.get("agents", AGENTS)),
                int(config.get("hidden_dim", 256)),
                ensemble_size=int(config.get("ensemble_size", 5)),
                graph_layers=int(config.get("graph_layers", 2)),
                learned_edge_dynamics=bool(config.get("learned_edge_dynamics", True)),
                edge_rank=int(config.get("edge_rank", 24)),
                edge_loss_coef=float(config.get("edge_loss_coef", 1.0)),
            ).to(device)
            world_model.load_state_dict(checkpoint["world_model_state_dict"])
            world_model.eval()
            model.cbg_world_model = world_model
            model.cbg_belief_config = {
                "max_age_s": float(config.get("belief_max_age_s", 3.0)),
                "covariance_growth": float(config.get("belief_covariance_growth", 0.08)),
                "sensor_delay_steps": int(config.get("sensor_delay_steps", 1)),
                "observation_dropout": float(config.get("observation_dropout", 0.05)),
                "uncertainty_enabled": bool(config.get("belief_uncertainty_enabled", True)),
            }
            if bool(config.get("mpc_enabled", True)):
                model.cbg_planner = FlowProposalRiskMPC(
                    world_model,
                    horizon=int(config.get("mpc_horizon", 5)),
                    candidates=int(config.get("mpc_candidates", 24)),
                    gamma=float(config.get("gamma", 0.995)),
                    cvar_beta=float(config.get("cvar_beta", 0.90)),
                    risk_coef=float(config.get("risk_coef", 2.0)),
                    uncertainty_coef=float(config.get("uncertainty_coef", 0.25)),
                    proposal_noise=float(config.get("proposal_noise", 0.12)),
                    particles_per_member=int(config.get("mpc_particles_per_member", 16)),
                    rollout_chunk_size=int(config.get("mpc_rollout_chunk_size", 64)),
                )
        model.eval()
        return model, checkpoint

    raise ValueError(
        f"unsupported checkpoint algorithm {algorithm!r}; "
        "the formal evaluator only accepts object-centric world-model SAC Flow checkpoints"
    )


def actor_action(
    model: torch.nn.Module,
    obs: np.ndarray,
    team: str,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
    team_index = 0 if team == "yellow" else 1
    with torch.no_grad():
        if not isinstance(model, MultiAgentFlowActors):
            raise TypeError("formal evaluation expects MultiAgentFlowActors")
        actor: FlowActor = model._actor(team_index)
        if deterministic:
            action = actor.deterministic(obs_t)
        else:
            action = actor.sample(obs_t)[0]
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run_episode(
    model: torch.nn.Module,
    *,
    seed: int,
    max_steps: int,
    device: torch.device,
    deterministic: bool,
    capture_trace: bool,
    policy_mode: str,
    residual_scale: float,
) -> dict[str, object]:
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=seed)
    rewards_total = {team: 0.0 for team in AGENTS}
    planner_metrics = {"cvar_return": [], "expected_risk": [], "epistemic_penalty": []}
    belief_tracker = None
    planner = getattr(model, "cbg_planner", None)
    if planner is not None:
        belief_tracker = BeliefTracker(seed=seed, **getattr(model, "cbg_belief_config", {}))
    events = {
        "normal_hits": 0,
        "base_hit_wins": 0,
        "own_target_penalties": 0,
        "collision_recovery_events": 0,
        "robot_contacts": 0,
        "block_steps": 0,
        "base_rush_steps": 0,
        "interference_steps": 0,
    }
    trace = []

    steps = 0
    for steps in range(1, max_steps + 1):
        actions: dict[str, np.ndarray] = {}
        raw_actions: dict[str, np.ndarray] = {}
        if planner is not None and belief_tracker is not None:
            joint_obs = torch.as_tensor(
                np.stack([observations[team] for team in AGENTS])[None],
                dtype=torch.float32,
                device=device,
            )
            belief = torch.as_tensor(
                belief_tracker.observe(env).tokens[None], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                plan = planner.plan(model, joint_obs, belief)
            for index, team in enumerate(AGENTS):
                raw_actions[team] = plan.actions[0, index].detach().cpu().numpy().astype(np.float32)
            planner_metrics["cvar_return"].append(float(plan.cvar_return.mean().cpu()))
            planner_metrics["expected_risk"].append(float(plan.expected_risk.mean().cpu()))
            planner_metrics["epistemic_penalty"].append(float(plan.epistemic_penalty.mean().cpu()))
        else:
            for team in AGENTS:
                raw_actions[team] = actor_action(model, observations[team], team, device, deterministic)
        for team in AGENTS:
            raw_action = raw_actions[team]
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
            if "hit" in info:
                events["normal_hits"] += 1
            if "winner" in info:
                events["base_hit_wins"] += 1
            if any(
                key in info
                for key in ("own_target_hit", "own_base_hit", "own_target_blocked", "own_base_blocked", "own_base_collision")
            ):
                events["own_target_penalties"] += 1
            if info.get("relocalizing"):
                events["collision_recovery_events"] += 1
            if info.get("robot_contact"):
                events["robot_contacts"] += 1
            if info.get("tactic") == "block":
                events["block_steps"] += 1
            if info.get("base_rush"):
                events["base_rush_steps"] += 1
            if info.get("interference"):
                events["interference_steps"] += 1

        if capture_trace and (steps % 20 == 0 or any("hit" in infos[team] or "winner" in infos[team] for team in AGENTS)):
            trace.append(
                {
                    "step": steps,
                    "elapsed_s": round(float(env.elapsed), 3),
                    "scores": dict(env.scores),
                    "armor": dict(env.armor),
                    "yellow_pose": [round(float(v), 4) for v in env.poses["yellow"]],
                    "blue_pose": [round(float(v), 4) for v in env.poses["blue"]],
                    "yellow_raw_action": [round(float(v), 4) for v in raw_actions["yellow"]],
                    "blue_raw_action": [round(float(v), 4) for v in raw_actions["blue"]],
                    "yellow_env_action": [round(float(v), 4) for v in actions["yellow"]],
                    "blue_env_action": [round(float(v), 4) for v in actions["blue"]],
                    "yellow_info": {k: v for k, v in infos["yellow"].items() if k != "action_labels"},
                    "blue_info": {k: v for k, v in infos["blue"].items() if k != "action_labels"},
                }
            )
        if any(terminations.values()) or any(truncations.values()):
            break

    return {
        "winner": env.winner or "timeout",
        "elapsed_s": round(float(env.elapsed), 3),
        "steps": steps,
        "scores": dict(env.scores),
        "armor": dict(env.armor),
        "rewards": {team: round(value, 3) for team, value in rewards_total.items()},
        "events": events,
        "target_order": {team: list(env.target_order[team]) for team in AGENTS},
        "strategy_counts": {team: dict(env.strategy_counts[team]) for team in AGENTS},
        "planner": {
            key: round(float(np.mean(values)), 6) if values else 0.0
            for key, values in planner_metrics.items()
        },
        "trace": trace,
    }


def summarize(episodes: list[dict[str, object]], wall_time_s: float) -> dict[str, object]:
    count = len(episodes)
    winners = [str(episode["winner"]) for episode in episodes]
    total_steps = sum(int(episode["steps"]) for episode in episodes)
    event_keys = episodes[0]["events"].keys()
    totals = {
        key: sum(int(episode["events"][key]) for episode in episodes)
        for key in event_keys
    }
    score_yellow = [int(episode["scores"]["yellow"]) for episode in episodes]
    score_blue = [int(episode["scores"]["blue"]) for episode in episodes]
    planner_keys = episodes[0].get("planner", {}).keys()
    return {
        "episodes": count,
        "yellow_win_rate": round(winners.count("yellow") / count, 4),
        "blue_win_rate": round(winners.count("blue") / count, 4),
        "draw_or_timeout_rate": round((winners.count("draw") + winners.count("timeout")) / count, 4),
        "mean_episode_time_s": round(sum(float(ep["elapsed_s"]) for ep in episodes) / count, 4),
        "mean_yellow_score": round(float(np.mean(score_yellow)), 4),
        "mean_blue_score": round(float(np.mean(score_blue)), 4),
        "normal_hits_per_episode": round(totals["normal_hits"] / count, 4),
        "base_hit_wins_per_episode": round(totals["base_hit_wins"] / count, 4),
        "own_target_penalties_per_episode": round(totals["own_target_penalties"] / count, 4),
        "collision_recovery_events_per_episode": round(totals["collision_recovery_events"] / count, 4),
        "robot_contacts_per_episode": round(totals["robot_contacts"] / count, 4),
        "block_steps_per_episode": round(totals["block_steps"] / count, 4),
        "base_rush_steps_per_episode": round(totals["base_rush_steps"] / count, 4),
        "interference_steps_per_episode": round(totals["interference_steps"] / count, 4),
        "planner": {
            key: round(float(np.mean([episode.get("planner", {}).get(key, 0.0) for episode in episodes])), 6)
            for key in planner_keys
        },
        "simulated_steps_per_second": round(total_steps / max(wall_time_s, 1e-9), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved tactical self-play policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--trace-episodes", type=int, default=4)
    parser.add_argument("--policy-mode", choices=("auto", "direct", "expert", "residual_expert"), default="auto")
    parser.add_argument("--residual-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("../output/eval/policy_eval.json"))
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
            capture_trace=index < args.trace_episodes,
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
        for index in range(args.episodes)
    ]
    wall_time_s = time.perf_counter() - started
    payload = {
        "summary": summarize(episodes, wall_time_s),
        "checkpoint": str(args.checkpoint),
        "deterministic": not args.stochastic,
        "policy_mode": policy_mode,
        "residual_scale": residual_scale,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "training_config": train_config,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"[INFO] wrote {args.output}")


if __name__ == "__main__":
    main()
