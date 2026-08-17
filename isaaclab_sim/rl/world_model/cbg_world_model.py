from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .belief_graph import (
    BELIEF_TOKEN_DIM,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    NUM_RULE_RISKS,
    PHYSICAL_TOKEN_DIM,
    TOKEN_AGE,
    TOKEN_COVARIANCE,
    TOKEN_LAST_SEEN,
    TOKEN_OCCLUDED,
    TOKEN_PRESENT,
    TOKEN_VISIBLE,
    build_typed_edges,
    canonical_node_types_torch,
)
from .constraint_graph_dynamics import (
    DURATION_BUCKET_COUNT,
    NUM_EDGE_EVENTS,
    ActionConditionedConstraintGraphDynamics,
    duration_bucket_targets,
    interventional_edge_consistency_loss,
)

LOGVAR_MIN = -7.0
LOGVAR_MAX = 2.0


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, hidden_dim), nn.SiLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class TypedInteractionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.relation_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(NUM_EDGE_TYPES)
        )
        self.update = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        relation_messages = []
        for relation, projection in enumerate(self.relation_projections):
            source = projection(hidden)
            adjacency = edges[:, relation]
            message = torch.einsum("bst,bsh->bth", adjacency, source)
            degree = adjacency.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
            relation_messages.append(message / degree)
        interaction = torch.stack(relation_messages, dim=0).sum(dim=0)
        updated = self.norm(hidden + self.update(torch.cat((hidden, interaction), dim=-1)))
        return updated, interaction


class TypedBeliefGraphEncoder(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, layers: int = 2):
        super().__init__()
        self.token_encoder = _mlp(token_dim, hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        self.layers = nn.ModuleList(TypedInteractionLayer(hidden_dim) for _ in range(max(int(layers), 0)))

    def forward(
        self,
        tokens: torch.Tensor,
        node_types: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.token_encoder(tokens) + self.type_embedding(node_types)
        hidden = base
        interaction_total = torch.zeros_like(base)
        for layer in self.layers:
            hidden, interaction = layer(hidden, edges)
            interaction_total = interaction_total + interaction
        return base, hidden, interaction_total


class BeliefGraphDynamicsMember(nn.Module):
    """One stochastic graph dynamics hypothesis in the CBG-WM ensemble."""

    def __init__(
        self,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
        *,
        graph_layers: int = 2,
        learned_edge_dynamics: bool = True,
        edge_rank: int = 24,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.graph_layers = max(int(graph_layers), 0)
        self.learned_edge_dynamics = bool(learned_edge_dynamics)
        self.edge_dynamics = (
            ActionConditionedConstraintGraphDynamics(
                action_dim,
                num_agents,
                hidden_dim,
                rank=edge_rank,
            )
            if self.learned_edge_dynamics
            else None
        )
        self.encoder = TypedBeliefGraphEncoder(BELIEF_TOKEN_DIM, hidden_dim, self.graph_layers)
        self.action_encoder = _mlp(action_dim * num_agents, hidden_dim, hidden_dim)
        self.self_dynamics = _mlp(hidden_dim * 2, hidden_dim, PHYSICAL_TOKEN_DIM)
        self.interaction_dynamics = _mlp(hidden_dim, hidden_dim, PHYSICAL_TOKEN_DIM)
        self.logvar_head = _mlp(hidden_dim * 2, hidden_dim, PHYSICAL_TOKEN_DIM)
        self.metadata_head = _mlp(hidden_dim, hidden_dim, 3)
        self.reward_head = _mlp(hidden_dim, hidden_dim, num_agents * 2)
        self.done_head = _mlp(hidden_dim, hidden_dim, num_agents)
        self.risk_head = _mlp(hidden_dim, hidden_dim, num_agents * NUM_RULE_RISKS)

    def _node_actions(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        encoded = self.action_encoder(actions.reshape(actions.shape[0], -1))
        return encoded.unsqueeze(1).expand(tokens.shape[0], tokens.shape[1], -1)

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        node_types: torch.Tensor | None = None,
        current_edges: torch.Tensor | None = None,
        edge_diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        if node_types is None:
            node_types = canonical_node_types_torch(tokens.shape[0], tokens.device)
        if current_edges is None:
            current_edges = build_typed_edges(tokens, node_types)
        if self.edge_dynamics is not None:
            edge_prediction = self.edge_dynamics(
                tokens, actions, node_types, current_edges, diagnostics=edge_diagnostics
            )
        else:
            valid = current_edges > -1.0
            edge_probability = current_edges.clamp(0.0, 1.0)
            presence_logits = torch.logit(edge_probability.clamp(1e-6, 1.0 - 1e-6))
            edge_prediction = {
                "presence_logits": presence_logits,
                "next_edge_prob": edge_probability,
                "current_edges": current_edges,
                "valid_mask": valid,
            }
            if edge_diagnostics:
                event_logits = torch.zeros(
                    *current_edges.shape,
                    NUM_EDGE_EVENTS,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                event_logits[..., 0] = 12.0
                edge_prediction.update(
                    {
                        "event_logits": event_logits,
                        "hazard_logits": torch.full_like(current_edges, -12.0),
                        "duration_logits": torch.zeros(
                            *current_edges.shape,
                            DURATION_BUCKET_COUNT,
                            dtype=tokens.dtype,
                            device=tokens.device,
                        ),
                    }
                )
        next_edges = edge_prediction["next_edge_prob"]
        base, hidden, interaction = self.encoder(tokens, node_types, next_edges)
        node_actions = self._node_actions(tokens, actions)
        self_delta = self.self_dynamics(torch.cat((base, node_actions), dim=-1))
        interaction_delta = self.interaction_dynamics(interaction)
        delta_mean = 0.16 * torch.tanh(self_delta + interaction_delta)
        logvar = self.logvar_head(torch.cat((hidden, node_actions), dim=-1)).clamp(LOGVAR_MIN, LOGVAR_MAX)

        next_tokens = tokens.clone()
        next_tokens[..., :PHYSICAL_TOKEN_DIM] = tokens[..., :PHYSICAL_TOKEN_DIM] + delta_mean
        metadata_logits = self.metadata_head(hidden)
        next_tokens[..., TOKEN_VISIBLE] = torch.sigmoid(metadata_logits[..., 0])
        next_tokens[..., TOKEN_OCCLUDED] = torch.sigmoid(metadata_logits[..., 1])
        next_tokens[..., TOKEN_PRESENT] = torch.sigmoid(metadata_logits[..., 2])
        next_tokens[..., TOKEN_AGE] = (tokens[..., TOKEN_AGE] + 0.04).clamp(0.0, 1.0)
        next_tokens[..., TOKEN_LAST_SEEN] = tokens[..., TOKEN_LAST_SEEN]
        next_tokens[..., TOKEN_COVARIANCE] = (
            tokens[..., TOKEN_COVARIANCE] + 0.02 * logvar.exp().mean(dim=-1)
        ).clamp(0.0, 1.0)

        present = tokens[..., TOKEN_PRESENT].clamp(0.0, 1.0).unsqueeze(-1)
        pooled = (hidden * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        reward_mean, reward_logvar = self.reward_head(pooled).chunk(2, dim=-1)
        reward_logvar = reward_logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
        result = {
            "next_tokens": next_tokens,
            "delta_mean": delta_mean,
            "state_logvar": logvar,
            "metadata_logits": metadata_logits,
            "reward_mean": reward_mean,
            "reward_logvar": reward_logvar,
            "done_logits": self.done_head(pooled),
            "risk_logits": self.risk_head(pooled).reshape(-1, self.num_agents, NUM_RULE_RISKS),
            "edges": current_edges,
            "next_edges": next_edges,
            "edge_presence_logits": edge_prediction["presence_logits"],
            "edge_valid_mask": edge_prediction["valid_mask"],
        }
        if edge_diagnostics:
            result.update(
                {
                    "edge_event_logits": edge_prediction["event_logits"],
                    "edge_hazard_logits": edge_prediction["hazard_logits"],
                    "edge_duration_logits": edge_prediction["duration_logits"],
                }
            )
        return result


class CounterfactualBeliefGraphWorldModel(nn.Module):
    """Probabilistic ensemble world model over typed object belief graphs."""

    def __init__(
        self,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
        *,
        ensemble_size: int = 5,
        graph_layers: int = 2,
        learned_edge_dynamics: bool = True,
        edge_rank: int = 24,
        edge_loss_coef: float = 1.0,
    ):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.ensemble_size = int(ensemble_size)
        self.graph_layers = max(int(graph_layers), 0)
        self.learned_edge_dynamics = bool(learned_edge_dynamics)
        self.edge_loss_coef = float(edge_loss_coef)
        self.members = nn.ModuleList(
            BeliefGraphDynamicsMember(
                action_dim,
                num_agents,
                hidden_dim,
                graph_layers=self.graph_layers,
                learned_edge_dynamics=learned_edge_dynamics,
                edge_rank=edge_rank,
            )
            for _ in range(ensemble_size)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        node_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        predictions = [member(tokens, actions, node_types) for member in self.members]
        keys = (
            "next_tokens",
            "delta_mean",
            "state_logvar",
            "metadata_logits",
            "reward_mean",
            "reward_logvar",
            "done_logits",
            "risk_logits",
            "edges",
            "next_edges",
            "edge_presence_logits",
            "edge_event_logits",
            "edge_hazard_logits",
            "edge_duration_logits",
            "edge_valid_mask",
        )
        return {key: torch.stack([prediction[key] for prediction in predictions], dim=0) for key in keys}

    def loss(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        next_tokens: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rule_risks: torch.Tensor,
        duration_targets: torch.Tensor | None = None,
        member_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        node_types = canonical_node_types_torch(tokens.shape[0], tokens.device)
        current_edges = build_typed_edges(tokens, node_types)
        next_edges = build_typed_edges(next_tokens, node_types)
        present_mask = torch.maximum(
            tokens[..., TOKEN_PRESENT], next_tokens[..., TOKEN_PRESENT]
        ).unsqueeze(-1)
        target_delta = next_tokens[..., :PHYSICAL_TOKEN_DIM] - tokens[..., :PHYSICAL_TOKEN_DIM]
        target_metadata = torch.stack(
            (
                next_tokens[..., TOKEN_VISIBLE],
                next_tokens[..., TOKEN_OCCLUDED],
                next_tokens[..., TOKEN_PRESENT],
            ),
            dim=-1,
        )

        member_losses = []
        state_nlls = []
        reward_nlls = []
        done_losses = []
        risk_losses = []
        metadata_losses = []
        edge_losses = []
        edge_metric_values: dict[str, list[torch.Tensor]] = {}
        means = []
        aleatoric = []
        selected_indices = range(self.ensemble_size) if member_indices is None else member_indices
        for member_index in selected_indices:
            member = self.members[member_index]
            prediction = member(tokens, actions, node_types, current_edges)
            if tokens.shape[0] > 1:
                bootstrap = torch.randint(0, tokens.shape[0], (tokens.shape[0],), device=tokens.device)
            else:
                bootstrap = torch.zeros(1, dtype=torch.long, device=tokens.device)
            state_error = target_delta[bootstrap] - prediction["delta_mean"][bootstrap]
            state_nll = 0.5 * (
                prediction["state_logvar"][bootstrap]
                + state_error.square() * torch.exp(-prediction["state_logvar"][bootstrap])
            )
            bootstrap_present = present_mask[bootstrap]
            state_nll = (state_nll * bootstrap_present).sum() / (
                bootstrap_present.sum() * PHYSICAL_TOKEN_DIM
            ).clamp_min(1.0)
            reward_error = rewards[bootstrap] - prediction["reward_mean"][bootstrap]
            reward_nll = 0.5 * (
                prediction["reward_logvar"][bootstrap]
                + reward_error.square() * torch.exp(-prediction["reward_logvar"][bootstrap])
            ).mean()
            done_loss = F.binary_cross_entropy_with_logits(
                prediction["done_logits"][bootstrap], dones[bootstrap]
            )
            risk_loss = F.binary_cross_entropy_with_logits(
                prediction["risk_logits"][bootstrap], rule_risks[bootstrap]
            )
            metadata_loss = F.binary_cross_entropy_with_logits(
                prediction["metadata_logits"][bootstrap], target_metadata[bootstrap]
            )
            edge_loss = prediction["edge_presence_logits"].sum() * 0.0
            if member.edge_dynamics is not None:
                edge_loss, edge_metrics = member.edge_dynamics.loss(
                    {key: value[bootstrap] for key, value in {
                        "presence_logits": prediction["edge_presence_logits"],
                        "event_logits": prediction["edge_event_logits"],
                        "hazard_logits": prediction["edge_hazard_logits"],
                        "duration_logits": prediction["edge_duration_logits"],
                        "current_edges": prediction["edges"],
                        "valid_mask": prediction["edge_valid_mask"],
                    }.items()},
                    next_edges[bootstrap],
                    duration_targets=duration_targets[bootstrap] if duration_targets is not None else None,
                )
                for key, value in edge_metrics.items():
                    edge_metric_values.setdefault(key, []).append(value)
            member_loss = (
                state_nll + 0.40 * reward_nll + 0.20 * done_loss + 0.35 * risk_loss + 0.10 * metadata_loss
                + self.edge_loss_coef * edge_loss
            )
            member_losses.append(member_loss)
            state_nlls.append(state_nll)
            reward_nlls.append(reward_nll)
            done_losses.append(done_loss)
            risk_losses.append(risk_loss)
            metadata_losses.append(metadata_loss)
            edge_losses.append(edge_loss)
            means.append(prediction["delta_mean"])
            aleatoric.append(prediction["state_logvar"].exp().mean())

        total = torch.stack(member_losses).mean()
        mean_stack = torch.stack(means, dim=0)
        state_rmse = torch.sqrt(
            (((mean_stack.mean(dim=0) - target_delta).square() * present_mask).sum())
            / (present_mask.sum() * PHYSICAL_TOKEN_DIM).clamp_min(1.0)
        )
        epistemic = mean_stack.var(dim=0, unbiased=False).mean()
        metrics = {
            "wm_state_nll": float(torch.stack(state_nlls).mean().detach().cpu()),
            "wm_state_rmse": float(state_rmse.detach().cpu()),
            "wm_reward_nll": float(torch.stack(reward_nlls).mean().detach().cpu()),
            "wm_done_loss": float(torch.stack(done_losses).mean().detach().cpu()),
            "wm_risk_loss": float(torch.stack(risk_losses).mean().detach().cpu()),
            "wm_metadata_loss": float(torch.stack(metadata_losses).mean().detach().cpu()),
            "wm_edge_loss": float(torch.stack(edge_losses).mean().detach().cpu()),
            "wm_aleatoric_var": float(torch.stack(aleatoric).mean().detach().cpu()),
            "wm_epistemic_var": float(epistemic.detach().cpu()),
        }
        for key, values in edge_metric_values.items():
            metrics[f"wm_{key}"] = float(torch.stack(values).mean().detach().cpu())
        return total, metrics

    def sequence_loss(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        next_tokens: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rule_risks: torch.Tensor,
        member_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train open-loop graph and object dynamics over an episode-safe sequence."""

        if tokens.ndim != 4 or actions.ndim != 4:
            raise ValueError("sequence tensors must have shape [batch, horizon, ...]")
        if tokens.shape[:2] != actions.shape[:2] or next_tokens.shape[:2] != actions.shape[:2]:
            raise ValueError("sequence batch and horizon dimensions must match")
        batch, horizon = actions.shape[:2]
        node_types_full = canonical_node_types_torch(batch, tokens.device)
        target_edge_steps = [build_typed_edges(tokens[:, 0], node_types_full)]
        target_edge_steps.extend(
            build_typed_edges(next_tokens[:, step], node_types_full) for step in range(horizon)
        )
        target_edge_sequence = torch.stack(target_edge_steps, dim=1)
        duration_targets = duration_bucket_targets(target_edge_sequence)[:, :horizon]

        member_losses = []
        state_losses = []
        reward_losses = []
        risk_losses = []
        edge_losses = []
        selected_indices = range(self.ensemble_size) if member_indices is None else member_indices
        for member_index in selected_indices:
            member = self.members[member_index]
            bootstrap = torch.randint(0, batch, (batch,), device=tokens.device) if batch > 1 else torch.zeros(1, dtype=torch.long, device=tokens.device)
            node_types = node_types_full[bootstrap]
            current_tokens = tokens[bootstrap, 0]
            current_edges = target_edge_sequence[bootstrap, 0]
            step_losses = []
            for step in range(horizon):
                prediction = member(
                    current_tokens,
                    actions[bootstrap, step],
                    node_types,
                    current_edges,
                )
                target_tokens = next_tokens[bootstrap, step]
                target_delta = target_tokens[..., :PHYSICAL_TOKEN_DIM] - current_tokens[..., :PHYSICAL_TOKEN_DIM]
                present_mask = torch.maximum(
                    current_tokens[..., TOKEN_PRESENT], target_tokens[..., TOKEN_PRESENT]
                ).unsqueeze(-1)
                state_error = target_delta - prediction["delta_mean"]
                state_nll = 0.5 * (
                    prediction["state_logvar"]
                    + state_error.square() * torch.exp(-prediction["state_logvar"])
                )
                state_nll = (state_nll * present_mask).sum() / (
                    present_mask.sum() * PHYSICAL_TOKEN_DIM
                ).clamp_min(1.0)
                reward_error = rewards[bootstrap, step] - prediction["reward_mean"]
                reward_nll = 0.5 * (
                    prediction["reward_logvar"]
                    + reward_error.square() * torch.exp(-prediction["reward_logvar"])
                ).mean()
                done_loss = F.binary_cross_entropy_with_logits(
                    prediction["done_logits"], dones[bootstrap, step]
                )
                risk_loss = F.binary_cross_entropy_with_logits(
                    prediction["risk_logits"], rule_risks[bootstrap, step]
                )
                target_metadata = torch.stack(
                    (
                        target_tokens[..., TOKEN_VISIBLE],
                        target_tokens[..., TOKEN_OCCLUDED],
                        target_tokens[..., TOKEN_PRESENT],
                    ),
                    dim=-1,
                )
                metadata_loss = F.binary_cross_entropy_with_logits(
                    prediction["metadata_logits"], target_metadata
                )
                edge_loss = prediction["edge_presence_logits"].sum() * 0.0
                if member.edge_dynamics is not None:
                    edge_loss, _edge_metrics = member.edge_dynamics.loss(
                        {
                            "presence_logits": prediction["edge_presence_logits"],
                            "event_logits": prediction["edge_event_logits"],
                            "hazard_logits": prediction["edge_hazard_logits"],
                            "duration_logits": prediction["edge_duration_logits"],
                            "current_edges": prediction["edges"],
                            "valid_mask": prediction["edge_valid_mask"],
                        },
                        target_edge_sequence[bootstrap, step + 1],
                        duration_targets=duration_targets[bootstrap, step],
                    )
                step_loss = (
                    state_nll
                    + 0.40 * reward_nll
                    + 0.20 * done_loss
                    + 0.35 * risk_loss
                    + 0.10 * metadata_loss
                    + self.edge_loss_coef * edge_loss
                )
                step_losses.append(step_loss)
                state_losses.append(state_nll.detach())
                reward_losses.append(reward_nll.detach())
                risk_losses.append(risk_loss.detach())
                edge_losses.append(edge_loss.detach())
                current_tokens = prediction["next_tokens"]
                current_edges = (
                    build_typed_edges(current_tokens, node_types)
                    if member.edge_dynamics is None and member.graph_layers > 0
                    else prediction["next_edges"]
                )
            member_losses.append(torch.stack(step_losses).mean())
        total = torch.stack(member_losses).mean()
        return total, {
            "wm_sequence_loss": float(total.detach().cpu()),
            "wm_sequence_state_nll": float(torch.stack(state_losses).mean().cpu()),
            "wm_sequence_reward_nll": float(torch.stack(reward_losses).mean().cpu()),
            "wm_sequence_risk_loss": float(torch.stack(risk_losses).mean().cpu()),
            "wm_sequence_edge_loss": float(torch.stack(edge_losses).mean().cpu()),
        }

    def paired_intervention_loss(
        self,
        factual_tokens: torch.Tensor,
        factual_actions: torch.Tensor,
        factual_next_tokens: torch.Tensor,
        factual_rewards: torch.Tensor,
        factual_risks: torch.Tensor,
        intervention_tokens: torch.Tensor,
        intervention_actions: torch.Tensor,
        intervention_next_tokens: torch.Tensor,
        intervention_rewards: torch.Tensor,
        intervention_risks: torch.Tensor,
        member_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Match predicted and simulator-observed effects between paired branches."""

        batch, horizon = factual_actions.shape[:2]
        if intervention_actions.shape[:2] != (batch, horizon):
            raise ValueError("paired branch batch and horizon dimensions must match")
        node_types = canonical_node_types_torch(batch, factual_tokens.device)
        member_losses = []
        edge_terms = []
        reward_terms = []
        risk_terms = []
        selected_indices = range(self.ensemble_size) if member_indices is None else member_indices
        for member_index in selected_indices:
            member = self.members[member_index]
            factual_state = factual_tokens[:, 0]
            intervention_state = intervention_tokens[:, 0]
            factual_edges = build_typed_edges(factual_state, node_types)
            intervention_edges = build_typed_edges(intervention_state, node_types)
            step_losses = []
            for step in range(horizon):
                factual_prediction = member(
                    factual_state,
                    factual_actions[:, step],
                    node_types,
                    factual_edges,
                )
                intervention_prediction = member(
                    intervention_state,
                    intervention_actions[:, step],
                    node_types,
                    intervention_edges,
                )
                factual_target_edges = build_typed_edges(
                    factual_next_tokens[:, step], node_types
                )
                intervention_target_edges = build_typed_edges(
                    intervention_next_tokens[:, step], node_types
                )
                valid = factual_prediction["edge_valid_mask"] | intervention_prediction["edge_valid_mask"]
                edge_term = interventional_edge_consistency_loss(
                    factual_prediction["next_edges"],
                    intervention_prediction["next_edges"],
                    factual_target_edges,
                    intervention_target_edges,
                    valid,
                )
                predicted_reward_effect = (
                    intervention_prediction["reward_mean"] - factual_prediction["reward_mean"]
                )
                target_reward_effect = intervention_rewards[:, step] - factual_rewards[:, step]
                reward_term = F.smooth_l1_loss(predicted_reward_effect, target_reward_effect)
                predicted_risk_effect = torch.sigmoid(intervention_prediction["risk_logits"]) - torch.sigmoid(
                    factual_prediction["risk_logits"]
                )
                target_risk_effect = intervention_risks[:, step] - factual_risks[:, step]
                risk_term = F.smooth_l1_loss(predicted_risk_effect, target_risk_effect)
                step_losses.append(edge_term + 0.5 * reward_term + 0.5 * risk_term)
                edge_terms.append(edge_term.detach())
                reward_terms.append(reward_term.detach())
                risk_terms.append(risk_term.detach())
                factual_state = factual_prediction["next_tokens"]
                intervention_state = intervention_prediction["next_tokens"]
                if member.edge_dynamics is None and member.graph_layers > 0:
                    factual_edges = build_typed_edges(factual_state, node_types)
                    intervention_edges = build_typed_edges(intervention_state, node_types)
                else:
                    factual_edges = factual_prediction["next_edges"]
                    intervention_edges = intervention_prediction["next_edges"]
            member_losses.append(torch.stack(step_losses).mean())
        total = torch.stack(member_losses).mean()
        return total, {
            "wm_intervention_loss": float(total.detach().cpu()),
            "wm_intervention_edge_loss": float(torch.stack(edge_terms).mean().cpu()),
            "wm_intervention_reward_loss": float(torch.stack(reward_terms).mean().cpu()),
            "wm_intervention_risk_loss": float(torch.stack(risk_terms).mean().cpu()),
        }

    @torch.no_grad()
    def rollout(
        self,
        initial_tokens: torch.Tensor,
        action_sequences: torch.Tensor,
        *,
        particles_per_member: int = 1,
        sample_state: bool = True,
        return_edges: bool = True,
        return_edge_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        if action_sequences.ndim != 4:
            raise ValueError(
                "action_sequences must have shape [batch, horizon, agents, action_dim]"
            )
        horizon = action_sequences.shape[1]
        particles = max(int(particles_per_member), 1)
        batch = initial_tokens.shape[0]
        expanded_tokens = initial_tokens.unsqueeze(0).expand(particles, *initial_tokens.shape).reshape(
            particles * batch, *initial_tokens.shape[1:]
        )
        expanded_actions = action_sequences.unsqueeze(0).expand(
            particles, *action_sequences.shape
        ).reshape(particles * batch, *action_sequences.shape[1:])
        member_states = [expanded_tokens.clone() for _ in self.members]
        node_types = canonical_node_types_torch(particles * batch, initial_tokens.device)
        member_edges = [build_typed_edges(expanded_tokens, node_types) for _ in self.members]
        trajectories = [torch.stack(member_states, dim=0)]
        edge_trajectories = [torch.stack(member_edges, dim=0)] if return_edges else []
        rewards = []
        dones = []
        risks = []
        sampled_risks = []
        aleatoric = []
        edge_events = []
        edge_hazards = []
        edge_durations = []
        for step in range(horizon):
            step_rewards = []
            step_dones = []
            step_risks = []
            step_aleatoric = []
            step_sampled_risks = []
            step_edge_events = []
            step_edge_hazards = []
            step_edge_durations = []
            next_member_states = []
            next_member_edges = []
            for member_index, member in enumerate(self.members):
                prediction = member(
                    member_states[member_index],
                    expanded_actions[:, step],
                    node_types,
                    member_edges[member_index],
                    edge_diagnostics=return_edge_diagnostics,
                )
                next_tokens = prediction["next_tokens"].clone()
                reward = prediction["reward_mean"]
                if sample_state:
                    state_noise = torch.randn_like(prediction["delta_mean"])
                    sampled_delta = prediction["delta_mean"] + state_noise * torch.exp(
                        0.5 * prediction["state_logvar"]
                    )
                    next_tokens[..., :PHYSICAL_TOKEN_DIM] = (
                        member_states[member_index][..., :PHYSICAL_TOKEN_DIM] + sampled_delta
                    ).clamp(-2.0, 2.0)
                    reward = reward + torch.randn_like(reward) * torch.exp(
                        0.5 * prediction["reward_logvar"]
                    )
                next_member_states.append(next_tokens)
                next_member_edges.append(
                    build_typed_edges(next_tokens, node_types)
                    if member.edge_dynamics is None and member.graph_layers > 0
                    else prediction["next_edges"]
                )
                step_rewards.append(reward)
                step_dones.append(torch.sigmoid(prediction["done_logits"]))
                risk_probability = torch.sigmoid(prediction["risk_logits"])
                step_risks.append(risk_probability)
                step_sampled_risks.append(
                    torch.bernoulli(risk_probability) if sample_state else risk_probability
                )
                step_aleatoric.append(prediction["state_logvar"].exp().mean(dim=(-1, -2)))
                if return_edge_diagnostics:
                    step_edge_events.append(prediction["edge_event_logits"])
                    step_edge_hazards.append(prediction["edge_hazard_logits"])
                    step_edge_durations.append(prediction["edge_duration_logits"])
            member_states = next_member_states
            member_edges = next_member_edges
            trajectories.append(torch.stack(member_states, dim=0))
            if return_edges:
                edge_trajectories.append(torch.stack(member_edges, dim=0))
            rewards.append(torch.stack(step_rewards, dim=0))
            dones.append(torch.stack(step_dones, dim=0))
            risks.append(torch.stack(step_risks, dim=0))
            sampled_risks.append(torch.stack(step_sampled_risks, dim=0))
            aleatoric.append(torch.stack(step_aleatoric, dim=0))
            if return_edge_diagnostics:
                edge_events.append(torch.stack(step_edge_events, dim=0))
                edge_hazards.append(torch.stack(step_edge_hazards, dim=0))
                edge_durations.append(torch.stack(step_edge_durations, dim=0))

        def flatten_samples(value: torch.Tensor) -> torch.Tensor:
            shaped = value.reshape(self.ensemble_size, particles, batch, *value.shape[2:])
            return shaped.reshape(self.ensemble_size * particles, batch, *value.shape[2:])

        result = {
            "tokens": flatten_samples(torch.stack(trajectories, dim=2)),
            "rewards": flatten_samples(torch.stack(rewards, dim=2)),
            "done_prob": flatten_samples(torch.stack(dones, dim=2)),
            "risk_prob": flatten_samples(torch.stack(risks, dim=2)),
            "risk_cost_sample": flatten_samples(torch.stack(sampled_risks, dim=2)),
            "aleatoric_var": flatten_samples(torch.stack(aleatoric, dim=2)),
            "ensemble_size": torch.as_tensor(self.ensemble_size, device=initial_tokens.device),
            "particles_per_member": torch.as_tensor(particles, device=initial_tokens.device),
        }
        if return_edges:
            result["edges"] = flatten_samples(torch.stack(edge_trajectories, dim=2))
        if return_edge_diagnostics:
            result["edge_event_logits"] = flatten_samples(torch.stack(edge_events, dim=2))
            result["edge_hazard_logits"] = flatten_samples(torch.stack(edge_hazards, dim=2))
            result["edge_duration_logits"] = flatten_samples(torch.stack(edge_durations, dim=2))
        return result

    @staticmethod
    def epistemic_disagreement(rollout: dict[str, torch.Tensor]) -> torch.Tensor:
        physical = rollout["tokens"][:, :, 1:, ..., :PHYSICAL_TOKEN_DIM]
        ensemble = int(rollout.get("ensemble_size", torch.as_tensor(physical.shape[0])).item())
        particles = int(rollout.get("particles_per_member", torch.as_tensor(1)).item())
        member_mean = physical.reshape(ensemble, particles, *physical.shape[1:]).mean(dim=1)
        return member_mean.var(dim=0, unbiased=False).mean(dim=(-1, -2))


def gaussian_coverage(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    sigma: float = 2.0,
) -> torch.Tensor:
    radius = float(sigma) * torch.exp(0.5 * logvar)
    return ((target >= mean - radius) & (target <= mean + radius)).float().mean()
