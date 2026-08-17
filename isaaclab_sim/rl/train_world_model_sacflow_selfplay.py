from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from expert_policy import compose_policy_action
from experiments.paired_interventions import add_pair_to_replay, generate_paired_intervention
from planning import FlowProposalRiskMPC
from policies import CentralizedTwinQ, FlowActor, ObjectWorldModel
from replay import EpisodeSequenceReplay
from robocup_visionrl_selfplay_env import AGENTS
from robocup_visionrl_selfplay_vec import RoboCupVisionRLSelfPlayVector
from torch import nn
from world_model import (
    BELIEF_STATE_DIM,
    NUM_RULE_RISKS,
    BeliefTracker,
    CounterfactualBeliefGraphWorldModel,
    extract_rule_risks,
    tokens_from_flat,
)
from world_model.object_state import OBJECT_STATE_DIM, extract_object_state


@dataclass
class TrainConfig:
    training_variant: str
    legacy_flat_state: bool
    timesteps: int
    num_envs: int
    env_workers: int
    seed: int
    hidden_dim: int
    batch_size: int
    replay_size: int
    learning_starts: int
    gradient_steps: int
    train_every_env_steps: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    world_model_lr: float
    alpha_lr: float
    target_entropy: float
    max_grad_norm: float
    flow_steps: int
    flow_velocity_scale: float
    actor_mode: str
    policy_mode: str
    residual_scale: float
    domain_randomization: bool
    action_shield: bool
    world_model_coef: float
    ensemble_size: int
    graph_layers: int
    learned_edge_dynamics: bool
    edge_rank: int
    edge_loss_coef: float
    belief_max_age_s: float
    belief_covariance_growth: float
    sensor_delay_steps: int
    observation_dropout: float
    belief_uncertainty_enabled: bool
    sequence_horizon: int
    sequence_batch_size: int
    world_model_members_per_update: int
    paired_batch_size: int
    paired_pairs: int
    paired_intervention_coef: float
    mpc_enabled: bool
    mpc_horizon: int
    mpc_candidates: int
    training_mpc_candidates: int
    training_mpc_interval_env_steps: int
    cvar_beta: float
    risk_coef: float
    uncertainty_coef: float
    proposal_noise: float
    mpc_particles_per_member: int
    mpc_rollout_chunk_size: int


def load_defaults(path: Path | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "training_variant": "full_accgd_cbg_wm",
        "legacy_flat_state": False,
        "timesteps": 100_000,
        "num_envs": 16,
        "env_workers": 0,
        "seed": 7,
        "hidden_dim": 256,
        "batch_size": 512,
        "replay_size": 200_000,
        "learning_starts": 2048,
        "gradient_steps": 1,
        "train_every_env_steps": 32,
        "gamma": 0.995,
        "tau": 0.01,
        "actor_lr": 3.0e-4,
        "critic_lr": 3.0e-4,
        "world_model_lr": 3.0e-4,
        "alpha_lr": 3.0e-4,
        "target_entropy": -6.0,
        "max_grad_norm": 1.0,
        "flow_steps": 3,
        "flow_velocity_scale": 0.20,
        "actor_mode": "dual",
        "policy_mode": "residual_expert",
        "residual_scale": 0.04,
        "domain_randomization": True,
        "action_shield": True,
        "world_model_coef": 0.25,
        "ensemble_size": 5,
        "graph_layers": 2,
        "learned_edge_dynamics": True,
        "edge_rank": 24,
        "edge_loss_coef": 1.0,
        "belief_max_age_s": 3.0,
        "belief_covariance_growth": 0.08,
        "sensor_delay_steps": 1,
        "observation_dropout": 0.05,
        "belief_uncertainty_enabled": True,
        "sequence_horizon": 10,
        "sequence_batch_size": 8,
        "world_model_members_per_update": 1,
        "paired_batch_size": 16,
        "paired_pairs": 128,
        "paired_intervention_coef": 0.5,
        "mpc_enabled": True,
        "mpc_horizon": 5,
        "mpc_candidates": 24,
        "training_mpc_candidates": 6,
        "training_mpc_interval_env_steps": 1024,
        "cvar_beta": 0.90,
        "risk_coef": 2.0,
        "uncertainty_coef": 0.25,
        "proposal_noise": 0.12,
        "mpc_particles_per_member": 16,
        "mpc_rollout_chunk_size": 64,
        "device": "auto",
        "output": "../output/rl/cbg_wm_selfplay",
    }
    if path is None:
        return defaults
    resolved = path if path.is_absolute() else Path(__file__).resolve().parent / path
    if not resolved.exists():
        return defaults
    config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    for key, value in config.items():
        if key in defaults:
            defaults[key] = value
    return defaults


def validate_variant_config(cfg: TrainConfig) -> None:
    variant = cfg.training_variant
    errors: list[str] = []
    if variant == "legacy_sac_flow":
        if not cfg.legacy_flat_state or cfg.mpc_enabled:
            errors.append("legacy_sac_flow requires legacy_flat_state=true and mpc_enabled=false")
    elif cfg.legacy_flat_state:
        errors.append(f"{variant} cannot use legacy_flat_state")

    if variant == "no_belief_uncertainty" and cfg.belief_uncertainty_enabled:
        errors.append("no_belief_uncertainty must zero only the explicit uncertainty fields")
    if variant == "no_interaction_graph":
        if cfg.graph_layers != 0 or cfg.learned_edge_dynamics:
            errors.append("no_interaction_graph requires graph_layers=0 and learned_edge_dynamics=false")
    if variant == "static_rule_graph":
        if cfg.graph_layers <= 0 or cfg.learned_edge_dynamics:
            errors.append("static_rule_graph requires message passing with learned_edge_dynamics=false")
    if variant == "dynamic_graph_no_pairs":
        if not cfg.learned_edge_dynamics or cfg.paired_intervention_coef != 0.0:
            errors.append("dynamic_graph_no_pairs requires learned edges and paired_intervention_coef=0")
    if variant == "full_accgd_cbg_wm":
        if not cfg.learned_edge_dynamics or cfg.paired_intervention_coef <= 0.0:
            errors.append("full_accgd_cbg_wm requires learned edges and paired intervention loss")
    if variant not in {
        "legacy_sac_flow",
        "no_belief_uncertainty",
        "no_interaction_graph",
        "static_rule_graph",
        "dynamic_graph_no_pairs",
        "full_accgd_cbg_wm",
    }:
        errors.append(f"unknown registered training variant: {variant}")
    if errors:
        raise ValueError("; ".join(errors))


class MultiAgentFlowActors(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        actor_mode: str,
        flow_steps: int,
        velocity_scale: float,
    ):
        super().__init__()
        self.actor_mode = actor_mode
        if actor_mode == "shared":
            self.shared_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
        elif actor_mode == "dual":
            self.yellow_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
            self.blue_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
        else:
            raise ValueError(f"unknown actor_mode: {actor_mode}")

    def _actor(self, index: int) -> FlowActor:
        if self.actor_mode == "shared":
            return self.shared_actor
        return self.yellow_actor if index == 0 else self.blue_actor

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = []
        log_probs = []
        for index in range(obs.shape[1]):
            action, log_prob, _raw = self._actor(index).sample(obs[:, index, :])
            actions.append(action)
            log_probs.append(log_prob)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        actions = [self._actor(index).deterministic(obs[:, index, :]) for index in range(obs.shape[1])]
        return torch.stack(actions, dim=1)


def observations_to_array(observations: list[dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack(
        [np.stack([np.asarray(obs[team], dtype=np.float32) for team in AGENTS]) for obs in observations]
    ).astype(np.float32)


def belief_states_to_array(
    vec: RoboCupVisionRLSelfPlayVector,
    trackers: list[BeliefTracker],
) -> np.ndarray:
    return np.stack(
        [tracker.observe(env).flatten() for env, tracker in zip(vec.envs, trackers)]
    ).astype(np.float32)


def object_states_to_array(vec: RoboCupVisionRLSelfPlayVector) -> np.ndarray:
    return np.stack([extract_object_state(env) for env in vec.envs]).astype(np.float32)


def actions_to_env(
    vec: RoboCupVisionRLSelfPlayVector,
    raw_actions: np.ndarray,
    *,
    policy_mode: str,
    residual_scale: float,
) -> list[dict[str, np.ndarray]]:
    clipped = np.clip(raw_actions, -1.0, 1.0).astype(np.float32)
    if hasattr(vec, "compose_actions"):
        return vec.compose_actions(
            clipped,
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
    action_dicts: list[dict[str, np.ndarray]] = []
    for env_index, env in enumerate(vec.envs):
        item = {}
        for team_index, team in enumerate(AGENTS):
            item[team] = compose_policy_action(
                env,
                team,
                clipped[env_index, team_index],
                policy_mode=policy_mode,
                residual_scale=residual_scale,
            )
        action_dicts.append(item)
    return action_dicts


def random_actions(num_envs: int, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(num_envs, len(AGENTS), action_dim)).astype(np.float32)


def rewards_to_array(rewards: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[item[team] for team in AGENTS] for item in rewards], dtype=np.float32)


def dones_to_array(terminations: list[dict[str, bool]], truncations: list[dict[str, bool]]) -> np.ndarray:
    return np.asarray(
        [
            [bool(terminations[index][team] or truncations[index][team]) for team in AGENTS]
            for index in range(len(terminations))
        ],
        dtype=np.float32,
    )


def rule_risks_to_array(
    infos: list[dict[str, dict[str, object]]],
    executed_actions: list[dict[str, np.ndarray]],
) -> np.ndarray:
    return np.stack(
        [extract_rule_risks(info, action) for info, action in zip(infos, executed_actions)]
    ).astype(np.float32)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)


def update_step(
    *,
    batch,
    sequence_batch,
    paired_batch,
    member_indices,
    actors: MultiAgentFlowActors,
    critic: CentralizedTwinQ,
    target_critic: CentralizedTwinQ,
    world_model: nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    world_model_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_actions, next_log_probs = actors.sample(batch.next_obs)
        target_q1, target_q2 = target_critic(batch.next_belief_state, batch.next_obs, next_actions)
        target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_probs
        backup = batch.rewards + cfg.gamma * (1.0 - batch.dones) * target_q

    current_q1, current_q2 = critic(batch.belief_state, batch.obs, batch.actions)
    critic_loss = (current_q1 - backup).pow(2).mean() + (current_q2 - backup).pow(2).mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
    critic_optimizer.step()

    new_actions, log_probs = actors.sample(batch.obs)
    q1_pi, q2_pi = critic(batch.belief_state, batch.obs, new_actions)
    q_pi = torch.minimum(q1_pi, q2_pi)
    actor_loss = (alpha * log_probs - q_pi).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actors.parameters(), cfg.max_grad_norm)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_probs.detach() + cfg.target_entropy)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()

    if cfg.legacy_flat_state:
        world_model_loss, world_metrics = world_model.loss(
            batch.belief_state,
            batch.actions,
            batch.next_belief_state,
            batch.rewards,
            batch.dones,
        )
    elif sequence_batch is None:
        world_model_loss, world_metrics = world_model.loss(
            tokens_from_flat(batch.belief_state),
            batch.actions,
            tokens_from_flat(batch.next_belief_state),
            batch.rewards,
            batch.dones,
            batch.rule_risks,
            member_indices=member_indices,
        )
    else:
        world_model_loss, world_metrics = world_model.sequence_loss(
            tokens_from_flat(sequence_batch.belief_state),
            sequence_batch.actions,
            tokens_from_flat(sequence_batch.next_belief_state),
            sequence_batch.rewards,
            sequence_batch.dones,
            sequence_batch.rule_risks,
            member_indices=member_indices,
        )
    intervention_loss = world_model_loss.new_zeros(())
    intervention_metrics: dict[str, float] = {}
    if not cfg.legacy_flat_state and paired_batch is not None and cfg.paired_intervention_coef > 0.0:
        intervention_loss, intervention_metrics = world_model.paired_intervention_loss(
            tokens_from_flat(paired_batch.factual.belief_state),
            paired_batch.factual.actions,
            tokens_from_flat(paired_batch.factual.next_belief_state),
            paired_batch.factual.rewards,
            paired_batch.factual.rule_risks,
            tokens_from_flat(paired_batch.intervention.belief_state),
            paired_batch.intervention.actions,
            tokens_from_flat(paired_batch.intervention.next_belief_state),
            paired_batch.intervention.rewards,
            paired_batch.intervention.rule_risks,
            member_indices=member_indices,
        )
    combined_world_model_loss = world_model_loss + cfg.paired_intervention_coef * intervention_loss
    weighted_world_model_loss = cfg.world_model_coef * combined_world_model_loss
    world_model_optimizer.zero_grad(set_to_none=True)
    weighted_world_model_loss.backward()
    nn.utils.clip_grad_norm_(world_model.parameters(), cfg.max_grad_norm)
    world_model_optimizer.step()

    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "alpha_loss": float(alpha_loss.detach().cpu()),
        "alpha": float(log_alpha.exp().detach().cpu()),
        "q_mean": float(q_pi.detach().mean().cpu()),
        **world_metrics,
        **intervention_metrics,
    }


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("configs/world_model_flow.yaml"))
    pre_args, _unknown = pre_parser.parse_known_args()
    defaults = load_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train the CBG-WM SAC Flow self-play policy.",
        parents=[pre_parser],
    )
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(flag, type=float, default=value)
        else:
            parser.add_argument(flag, type=str, default=value)
    args = parser.parse_args()

    cfg = TrainConfig(
        training_variant=str(args.training_variant),
        legacy_flat_state=bool(args.legacy_flat_state),
        timesteps=int(args.timesteps),
        num_envs=int(args.num_envs),
        env_workers=int(args.env_workers),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        replay_size=int(args.replay_size),
        learning_starts=int(args.learning_starts),
        gradient_steps=int(args.gradient_steps),
        train_every_env_steps=int(args.train_every_env_steps),
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        world_model_lr=float(args.world_model_lr),
        alpha_lr=float(args.alpha_lr),
        target_entropy=float(args.target_entropy),
        max_grad_norm=float(args.max_grad_norm),
        flow_steps=int(args.flow_steps),
        flow_velocity_scale=float(args.flow_velocity_scale),
        actor_mode=str(args.actor_mode),
        policy_mode=str(args.policy_mode),
        residual_scale=float(args.residual_scale),
        domain_randomization=bool(args.domain_randomization),
        action_shield=bool(args.action_shield),
        world_model_coef=float(args.world_model_coef),
        ensemble_size=int(args.ensemble_size),
        graph_layers=int(args.graph_layers),
        learned_edge_dynamics=bool(args.learned_edge_dynamics),
        edge_rank=int(args.edge_rank),
        edge_loss_coef=float(args.edge_loss_coef),
        belief_max_age_s=float(args.belief_max_age_s),
        belief_covariance_growth=float(args.belief_covariance_growth),
        sensor_delay_steps=int(args.sensor_delay_steps),
        observation_dropout=float(args.observation_dropout),
        belief_uncertainty_enabled=bool(args.belief_uncertainty_enabled),
        sequence_horizon=int(args.sequence_horizon),
        sequence_batch_size=int(args.sequence_batch_size),
        world_model_members_per_update=int(args.world_model_members_per_update),
        paired_batch_size=int(args.paired_batch_size),
        paired_pairs=int(args.paired_pairs),
        paired_intervention_coef=float(args.paired_intervention_coef),
        mpc_enabled=bool(args.mpc_enabled),
        mpc_horizon=int(args.mpc_horizon),
        mpc_candidates=int(args.mpc_candidates),
        training_mpc_candidates=int(args.training_mpc_candidates),
        training_mpc_interval_env_steps=int(args.training_mpc_interval_env_steps),
        cvar_beta=float(args.cvar_beta),
        risk_coef=float(args.risk_coef),
        uncertainty_coef=float(args.uncertainty_coef),
        proposal_noise=float(args.proposal_noise),
        mpc_particles_per_member=int(args.mpc_particles_per_member),
        mpc_rollout_chunk_size=int(args.mpc_rollout_chunk_size),
    )
    validate_variant_config(cfg)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    output_dir = (Path(__file__).resolve().parent / str(args.output)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curve.csv"
    summary_path = output_dir / "training_summary.json"

    vec = RoboCupVisionRLSelfPlayVector(
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        env_kwargs={
            "domain_randomization": cfg.domain_randomization,
            "action_shield": cfg.action_shield,
        },
        workers=cfg.env_workers,
    )
    observations, _infos = vec.reset()
    obs_array = observations_to_array(observations)
    belief_trackers = [
        BeliefTracker(
            max_age_s=cfg.belief_max_age_s,
            covariance_growth=cfg.belief_covariance_growth,
            sensor_delay_steps=cfg.sensor_delay_steps,
            observation_dropout=cfg.observation_dropout,
            uncertainty_enabled=cfg.belief_uncertainty_enabled,
            seed=cfg.seed + index,
        )
        for index in range(cfg.num_envs)
    ]
    belief_array = (
        object_states_to_array(vec)
        if cfg.legacy_flat_state
        else belief_states_to_array(vec, belief_trackers)
    )
    state_dim = OBJECT_STATE_DIM if cfg.legacy_flat_state else BELIEF_STATE_DIM
    obs_dim = obs_array.shape[-1]
    action_dim = vec.envs[0].action_spaces["yellow"].shape[0]

    actors = MultiAgentFlowActors(
        obs_dim,
        action_dim,
        cfg.hidden_dim,
        actor_mode=cfg.actor_mode,
        flow_steps=cfg.flow_steps,
        velocity_scale=cfg.flow_velocity_scale,
    ).to(device)
    critic = CentralizedTwinQ(state_dim, obs_dim, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
    target_critic = CentralizedTwinQ(state_dim, obs_dim, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    if cfg.legacy_flat_state:
        world_model = ObjectWorldModel(state_dim, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
        planner = None
    else:
        world_model = CounterfactualBeliefGraphWorldModel(
            action_dim,
            len(AGENTS),
            cfg.hidden_dim,
            ensemble_size=cfg.ensemble_size,
            graph_layers=cfg.graph_layers,
            learned_edge_dynamics=cfg.learned_edge_dynamics,
            edge_rank=cfg.edge_rank,
            edge_loss_coef=cfg.edge_loss_coef,
        ).to(device)
        planner = FlowProposalRiskMPC(
            world_model,
            horizon=cfg.mpc_horizon,
            candidates=cfg.training_mpc_candidates,
            gamma=cfg.gamma,
            cvar_beta=cfg.cvar_beta,
            risk_coef=cfg.risk_coef,
            uncertainty_coef=cfg.uncertainty_coef,
            proposal_noise=cfg.proposal_noise,
            particles_per_member=cfg.mpc_particles_per_member,
            rollout_chunk_size=cfg.mpc_rollout_chunk_size,
        )
    actor_optimizer = torch.optim.Adam(actors.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    log_alpha = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    replay = EpisodeSequenceReplay(
        cfg.replay_size,
        len(AGENTS),
        obs_dim,
        state_dim,
        action_dim,
        NUM_RULE_RISKS,
        num_envs=cfg.num_envs,
        seed=cfg.seed,
    )
    paired_replay = EpisodeSequenceReplay(
        max(cfg.paired_pairs * 2 * cfg.sequence_horizon + 32, 64),
        len(AGENTS),
        obs_dim,
        state_dim,
        action_dim,
        NUM_RULE_RISKS,
        num_envs=1,
        seed=cfg.seed + 700_000,
    )
    if not cfg.legacy_flat_state and cfg.paired_intervention_coef > 0.0 and cfg.paired_pairs > 0:
        tracker_kwargs = {
            "max_age_s": cfg.belief_max_age_s,
            "covariance_growth": cfg.belief_covariance_growth,
            "sensor_delay_steps": cfg.sensor_delay_steps,
            "observation_dropout": cfg.observation_dropout,
            "uncertainty_enabled": cfg.belief_uncertainty_enabled,
        }
        for pair_index in range(cfg.paired_pairs):
            mechanism = "push_box" if pair_index % 2 == 0 else "remove_armor"
            pair = generate_paired_intervention(
                seed=cfg.seed + 800_000 + pair_index,
                mechanism=mechanism,
                horizon=cfg.sequence_horizon,
                tracker_kwargs=tracker_kwargs,
            )
            add_pair_to_replay(paired_replay, pair, episode_base=2 * pair_index)

    curve_file = curve_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        curve_file,
        fieldnames=[
            "env_step",
            "buffer_size",
            "mean_reward",
            "done_rate",
            "critic_loss",
            "actor_loss",
            "alpha",
            "q_mean",
            "wm_state_nll",
            "wm_state_rmse",
            "wm_reward_nll",
            "wm_done_loss",
            "wm_risk_loss",
            "wm_epistemic_var",
            "wm_sequence_edge_loss",
            "wm_intervention_loss",
            "mpc_cvar_return",
            "mpc_expected_risk",
            "steps_per_second",
        ],
    )
    writer.writeheader()
    started = time.perf_counter()
    update_metrics: dict[str, float] = {}
    next_reset_seed = cfg.seed + 100_000
    episode_seeds = [cfg.seed + index for index in range(cfg.num_envs)]
    gradient_update_count = 0
    world_model_member_cursor = 0

    try:
        env_step = 0
        while env_step < cfg.timesteps:
            if len(replay) < cfg.learning_starts:
                raw_actions = random_actions(cfg.num_envs, action_dim, rng)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
                    use_training_mpc = (
                        cfg.mpc_enabled
                        and planner is not None
                        and env_step % max(cfg.training_mpc_interval_env_steps, cfg.num_envs) == 0
                    )
                    if use_training_mpc:
                        belief_t = tokens_from_flat(
                            torch.as_tensor(belief_array, dtype=torch.float32, device=device)
                        )
                        mpc_result = planner.plan(actors, obs_t, belief_t)
                        raw_actions = mpc_result.actions.detach().cpu().numpy().astype(np.float32)
                        update_metrics["mpc_cvar_return"] = float(mpc_result.cvar_return.mean().cpu())
                        update_metrics["mpc_expected_risk"] = float(mpc_result.expected_risk.mean().cpu())
                    else:
                        raw_actions = actors.sample(obs_t)[0].detach().cpu().numpy().astype(np.float32)

            executed_actions = actions_to_env(
                vec, raw_actions, policy_mode=cfg.policy_mode, residual_scale=cfg.residual_scale
            )
            next_observations, rewards, terminations, truncations, infos = vec.step(
                executed_actions
            )
            reward_array = rewards_to_array(rewards)
            done_array = dones_to_array(terminations, truncations)
            risk_array = rule_risks_to_array(infos, executed_actions)
            next_obs_array_before_reset = observations_to_array(next_observations)
            next_belief_array_before_reset = (
                object_states_to_array(vec)
                if cfg.legacy_flat_state
                else belief_states_to_array(vec, belief_trackers)
            )

            for index in range(cfg.num_envs):
                replay.add(
                    obs_array[index],
                    belief_array[index],
                    raw_actions[index],
                    reward_array[index],
                    next_obs_array_before_reset[index],
                    next_belief_array_before_reset[index],
                    done_array[index],
                    risk_array[index],
                    env_id=index,
                    exogenous_seed=episode_seeds[index],
                )

            for index in range(cfg.num_envs):
                if bool(done_array[index].max() > 0.0):
                    episode_seeds[index] = next_reset_seed
                    next_observations[index], _ = vec.reset_one(index, seed=episode_seeds[index])
                    if cfg.legacy_flat_state:
                        next_belief_array_before_reset[index] = extract_object_state(vec.envs[index])
                    else:
                        belief_trackers[index].reset()
                        next_belief_array_before_reset[index] = belief_trackers[index].observe(vec.envs[index]).flatten()
                    next_reset_seed += 1

            observations = next_observations
            obs_array = observations_to_array(observations)
            belief_array = next_belief_array_before_reset
            env_step += cfg.num_envs

            should_train = (
                len(replay) >= cfg.learning_starts
                and env_step % max(cfg.train_every_env_steps, cfg.num_envs) == 0
            )
            if should_train:
                for _ in range(cfg.gradient_steps):
                    batch = replay.sample(cfg.batch_size, device)
                    try:
                        sequence_batch = replay.sample_sequences(
                            cfg.sequence_batch_size,
                            cfg.sequence_horizon,
                            device,
                        )
                    except RuntimeError:
                        sequence_batch = None
                    paired_batch = None
                    if cfg.paired_intervention_coef > 0.0 and len(paired_replay) > 0:
                        try:
                            paired_batch = paired_replay.sample_paired_sequences(
                                cfg.paired_batch_size,
                                cfg.sequence_horizon,
                                device,
                            )
                        except RuntimeError:
                            paired_batch = None
                    if cfg.legacy_flat_state:
                        member_indices = None
                    else:
                        member_count = min(
                            max(cfg.world_model_members_per_update, 1),
                            cfg.ensemble_size,
                        )
                        member_indices = [
                            (world_model_member_cursor + offset) % cfg.ensemble_size
                            for offset in range(member_count)
                        ]
                        world_model_member_cursor = (
                            world_model_member_cursor + member_count
                        ) % cfg.ensemble_size
                    update_metrics = update_step(
                        batch=batch,
                        sequence_batch=sequence_batch,
                        paired_batch=paired_batch,
                        member_indices=member_indices,
                        actors=actors,
                        critic=critic,
                        target_critic=target_critic,
                        world_model=world_model,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        world_model_optimizer=world_model_optimizer,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        cfg=cfg,
                    )
                    soft_update(critic, target_critic, cfg.tau)
                    gradient_update_count += 1

            if env_step % max(cfg.num_envs * 10, 1) == 0 or env_step >= cfg.timesteps:
                elapsed = max(time.perf_counter() - started, 1e-9)
                row = {
                    "env_step": env_step,
                    "buffer_size": len(replay),
                    "mean_reward": float(reward_array.mean()),
                    "done_rate": float(done_array.mean()),
                    "critic_loss": update_metrics.get("critic_loss", 0.0),
                    "actor_loss": update_metrics.get("actor_loss", 0.0),
                    "alpha": update_metrics.get("alpha", float(log_alpha.exp().detach().cpu())),
                    "q_mean": update_metrics.get("q_mean", 0.0),
                    "wm_state_nll": update_metrics.get("wm_state_nll", 0.0),
                    "wm_state_rmse": update_metrics.get("wm_state_rmse", 0.0),
                    "wm_reward_nll": update_metrics.get("wm_reward_nll", 0.0),
                    "wm_done_loss": update_metrics.get("wm_done_loss", 0.0),
                    "wm_risk_loss": update_metrics.get("wm_risk_loss", 0.0),
                    "wm_epistemic_var": update_metrics.get("wm_epistemic_var", 0.0),
                    "wm_sequence_edge_loss": update_metrics.get("wm_sequence_edge_loss", 0.0),
                    "wm_intervention_loss": update_metrics.get("wm_intervention_loss", 0.0),
                    "mpc_cvar_return": update_metrics.get("mpc_cvar_return", 0.0),
                    "mpc_expected_risk": update_metrics.get("mpc_expected_risk", 0.0),
                    "steps_per_second": float(env_step / elapsed),
                }
                writer.writerow(row)
                curve_file.flush()
                print(
                    "[WM-SACFLOW]: "
                    f"steps={env_step} buffer={len(replay)} reward={row['mean_reward']:.3f} "
                    f"done={row['done_rate']:.3f} q={row['q_mean']:.3f} alpha={row['alpha']:.3f}",
                    flush=True,
                )
    finally:
        curve_file.close()
        vec.close()

    algorithm = (
        "object_centric_world_model_sac_flow_selfplay"
        if cfg.legacy_flat_state
        else "cbg_wm_sac_flow_selfplay"
    )
    checkpoint = {
        "algorithm": algorithm,
        "training_variant": cfg.training_variant,
        "actor_state_dict": actors.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "world_model_state_dict": world_model.state_dict(),
        "log_alpha": float(log_alpha.detach().cpu()),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "belief_state_dim": state_dim if not cfg.legacy_flat_state else 0,
        "object_state_dim": state_dim,
        "action_dim": action_dim,
        "agents": list(AGENTS),
        "actor_mode": cfg.actor_mode,
    }
    policy_path = output_dir / "policy.pt"
    torch.save(checkpoint, policy_path)
    summary = {
        "algorithm": checkpoint["algorithm"],
        "policy_path": str(policy_path),
        "curve_csv": str(curve_path),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "belief_state_dim": state_dim if not cfg.legacy_flat_state else 0,
        "object_state_dim": state_dim,
        "action_dim": action_dim,
        "agents": list(AGENTS),
        "device": str(device),
        "torch_version": torch.__version__,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "gradient_update_count": gradient_update_count,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO]: Saved CBG-WM SAC Flow policy to {policy_path}", flush=True)


if __name__ == "__main__":
    main()
