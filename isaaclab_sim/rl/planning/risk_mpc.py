from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from world_model.cbg_world_model import CounterfactualBeliefGraphWorldModel


def cvar_lower_tail(values: torch.Tensor, alpha: float, dim: int = 0) -> torch.Tensor:
    """Mean of the worst ``alpha`` fraction along ``dim``."""

    alpha = float(min(max(alpha, 1e-6), 1.0))
    tail_count = max(1, math.ceil(values.shape[dim] * alpha))
    sorted_values = torch.sort(values, dim=dim).values
    return sorted_values.narrow(dim, 0, tail_count).mean(dim=dim)


def cvar_upper_tail(values: torch.Tensor, beta: float, dim: int = 0) -> torch.Tensor:
    """Mean of the highest-cost ``1 - beta`` fraction along ``dim``."""

    beta = float(min(max(beta, 0.0), 1.0 - 1e-6))
    tail_count = max(1, math.ceil(values.shape[dim] * (1.0 - beta)))
    sorted_values = torch.sort(values, dim=dim, descending=True).values
    return sorted_values.narrow(dim, 0, tail_count).mean(dim=dim)


@dataclass
class MPCResult:
    actions: torch.Tensor
    candidate_indices: torch.Tensor
    scores: torch.Tensor
    expected_return: torch.Tensor
    cvar_return: torch.Tensor
    expected_risk: torch.Tensor
    cvar_cost: torch.Tensor
    epistemic_penalty: torch.Tensor


class FlowProposalRiskMPC:
    """Short-horizon risk MPC using Flow actors as the trajectory prior."""

    def __init__(
        self,
        world_model: CounterfactualBeliefGraphWorldModel,
        *,
        horizon: int = 5,
        candidates: int = 32,
        gamma: float = 0.98,
        cvar_beta: float = 0.90,
        cvar_alpha: float | None = None,
        risk_coef: float = 2.0,
        risk_budgets: tuple[float, ...] | None = None,
        uncertainty_coef: float = 0.25,
        calibration_margin: float = 0.0,
        proposal_noise: float = 0.12,
        particles_per_member: int = 16,
        rollout_chunk_size: int = 64,
    ):
        self.world_model = world_model
        self.horizon = max(int(horizon), 1)
        self.candidates = max(int(candidates), 2)
        self.gamma = float(gamma)
        self.cvar_beta = float(1.0 - cvar_alpha) if cvar_alpha is not None else float(cvar_beta)
        self.risk_coef = float(risk_coef)
        self.risk_budgets = risk_budgets
        self.uncertainty_coef = float(uncertainty_coef)
        self.calibration_margin = max(float(calibration_margin), 0.0)
        self.proposal_noise = max(float(proposal_noise), 0.0)
        self.particles_per_member = max(int(particles_per_member), 1)
        self.rollout_chunk_size = max(int(rollout_chunk_size), 1)

    @torch.no_grad()
    def propose(self, actors, observations: torch.Tensor) -> torch.Tensor:
        batch, agents, obs_dim = observations.shape
        action_dim = actors._actor(0).action_dim
        opponent_hypothesis = actors.deterministic(observations)
        proposals = opponent_hypothesis[:, None, None, None].expand(
            batch,
            agents,
            self.candidates,
            self.horizon,
            agents,
            action_dim,
        ).clone()
        for ego in range(agents):
            expanded_ego_obs = observations[:, ego].unsqueeze(1).expand(
                batch, self.candidates, obs_dim
            ).reshape(batch * self.candidates, obs_dim)
            for step in range(self.horizon):
                sampled, _log_prob, _raw = actors._actor(ego).sample(expanded_ego_obs)
                sampled = sampled.reshape(batch, self.candidates, action_dim)
                if step == 0:
                    sampled[:, 0] = opponent_hypothesis[:, ego]
                if self.proposal_noise > 0.0 and step > 0:
                    noise = torch.randn_like(sampled) * self.proposal_noise * math.sqrt(
                        step / self.horizon
                    )
                    sampled = (sampled + noise).clamp(-1.0, 1.0)
                proposals[:, ego, :, step, ego] = sampled
        return proposals

    def _chunked_rollout(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensor_keys = (
            "tokens",
            "rewards",
            "done_prob",
            "risk_prob",
            "risk_cost_sample",
            "aleatoric_var",
        )
        chunks: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
        metadata = None
        for start in range(0, tokens.shape[0], self.rollout_chunk_size):
            stop = min(start + self.rollout_chunk_size, tokens.shape[0])
            rollout = self.world_model.rollout(
                tokens[start:stop],
                actions[start:stop],
                particles_per_member=self.particles_per_member,
                sample_state=True,
                return_edges=False,
            )
            for key in tensor_keys:
                chunks[key].append(rollout[key])
            if metadata is None:
                metadata = (rollout["ensemble_size"], rollout["particles_per_member"])
        merged = {key: torch.cat(value, dim=1) for key, value in chunks.items()}
        merged["ensemble_size"], merged["particles_per_member"] = metadata
        return merged

    @torch.no_grad()
    def plan(
        self,
        actors,
        observations: torch.Tensor,
        belief_tokens: torch.Tensor,
    ) -> MPCResult:
        batch, candidates, horizon, agents, action_dim = (
            observations.shape[0],
            self.candidates,
            self.horizon,
            observations.shape[1],
            actors._actor(0).action_dim,
        )
        proposals = self.propose(actors, observations)
        flat_tokens = (
            belief_tokens[:, None, None]
            .expand(batch, agents, candidates, *belief_tokens.shape[1:])
            .reshape(batch * agents * candidates, *belief_tokens.shape[1:])
        )
        flat_actions = proposals.reshape(
            batch * agents * candidates, horizon, agents, action_dim
        )
        rollout = self._chunked_rollout(flat_tokens, flat_actions)

        samples = rollout["rewards"].shape[0]
        rewards = rollout["rewards"].reshape(
            samples, batch, agents, candidates, horizon, agents
        )
        sampled_costs = rollout["risk_cost_sample"].reshape(
            samples, batch, agents, candidates, horizon, agents, -1
        )
        done = rollout["done_prob"].reshape(
            samples, batch, agents, candidates, horizon, agents
        )
        ego_rewards = torch.stack(
            [rewards[:, :, ego, :, :, ego] for ego in range(agents)], dim=2
        )
        ego_costs = torch.stack(
            [sampled_costs[:, :, ego, :, :, ego] for ego in range(agents)], dim=2
        )
        ego_done = torch.stack(
            [done[:, :, ego, :, :, ego] for ego in range(agents)], dim=2
        )
        discounts = torch.pow(
            torch.as_tensor(self.gamma, dtype=ego_rewards.dtype, device=ego_rewards.device),
            torch.arange(horizon, dtype=ego_rewards.dtype, device=ego_rewards.device),
        ).view(1, 1, 1, 1, horizon)
        survival = torch.cumprod(1.0 - ego_done.clamp(0.0, 1.0), dim=4)
        survival = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=4)
        sample_return = (discounts * survival * ego_rewards).sum(dim=4)
        sample_cost = (discounts.unsqueeze(-1) * ego_costs).sum(dim=4)
        expected_return = sample_return.mean(dim=0)
        expected_risk = sample_cost.mean(dim=0).sum(dim=-1)
        cvar_return = cvar_lower_tail(sample_return, 1.0 - self.cvar_beta, dim=0)
        cvar_cost = cvar_upper_tail(sample_cost, self.cvar_beta, dim=0)

        risk_channels = cvar_cost.shape[-1]
        budgets = torch.zeros(risk_channels, dtype=cvar_cost.dtype, device=cvar_cost.device)
        if self.risk_budgets is not None:
            if len(self.risk_budgets) != risk_channels:
                raise ValueError("risk_budgets must contain one value per rule-risk channel")
            budgets = torch.as_tensor(self.risk_budgets, dtype=cvar_cost.dtype, device=cvar_cost.device)
        risk_penalty = self.risk_coef * torch.relu(
            cvar_cost + self.calibration_margin - budgets
        ).sum(dim=-1)

        disagreement = self.world_model.epistemic_disagreement(rollout)
        disagreement = disagreement.reshape(batch, agents, candidates, horizon)
        epistemic_penalty = (
            disagreement
            * discounts.reshape(horizon).view(1, 1, 1, horizon)
        ).sum(dim=3)
        scores = expected_return - risk_penalty - self.uncertainty_coef * epistemic_penalty
        candidate_indices = scores.argmax(dim=2)

        actions = torch.empty(batch, agents, action_dim, dtype=proposals.dtype, device=proposals.device)
        batch_indices = torch.arange(batch, device=candidate_indices.device)
        for ego in range(agents):
            actions[:, ego] = proposals[
                batch_indices,
                ego,
                candidate_indices[:, ego],
                0,
                ego,
            ]
        actions = actions.clamp(-1.0, 1.0)
        return MPCResult(
            actions=actions,
            candidate_indices=candidate_indices,
            scores=scores,
            expected_return=expected_return,
            cvar_return=cvar_return,
            expected_risk=expected_risk,
            cvar_cost=cvar_cost,
            epistemic_penalty=epistemic_penalty,
        )
