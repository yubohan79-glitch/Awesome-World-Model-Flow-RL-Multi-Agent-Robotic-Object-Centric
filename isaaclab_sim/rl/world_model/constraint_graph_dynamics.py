from __future__ import annotations

import math
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import nn

from .belief_graph import (
    BELIEF_TOKEN_DIM,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    TOKEN_PRESENT,
    EdgeType,
    NodeType,
    build_typed_edges,
    canonical_node_types_torch,
)


class EdgeEvent(IntEnum):
    STAY = 0
    ADD = 1
    DELETE = 2


NUM_EDGE_EVENTS = len(EdgeEvent)
DURATION_BUCKET_COUNT = 4
OUTPUT_HEAD_COUNT = 1 + NUM_EDGE_EVENTS + 1 + DURATION_BUCKET_COUNT


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def typed_edge_valid_mask(
    node_types: torch.Tensor,
    present: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return valid relation slots with shape [batch, relation, source, target]."""

    if node_types.ndim == 1:
        node_types = node_types.unsqueeze(0)
    batch, nodes = node_types.shape
    source = node_types[:, :, None]
    target = node_types[:, None, :]
    not_self = ~torch.eye(nodes, dtype=torch.bool, device=node_types.device).unsqueeze(0)
    mask = torch.zeros(
        batch,
        NUM_EDGE_TYPES,
        nodes,
        nodes,
        dtype=torch.bool,
        device=node_types.device,
    )

    global_node = int(NodeType.GLOBAL)
    robot = int(NodeType.ROBOT)
    target_node = int(NodeType.TARGET)
    box = int(NodeType.BOX)
    armor = int(NodeType.ARMOR_BLOCKER)

    mask[:, int(EdgeType.GLOBAL)] = ((source == global_node) | (target == global_node)) & not_self
    mask[:, int(EdgeType.OBSERVES)] = (source == robot) & (target != global_node) & not_self
    mask[:, int(EdgeType.CONTACTS)] = (
        ((source == robot) & (target == box)) | ((source == box) & (target == robot))
    )
    mask[:, int(EdgeType.BLOCKS_ROUTE)] = (source == box) & (target == target_node)
    mask[:, int(EdgeType.PROTECTS_BASE)] = (source == armor) & (target == target_node)
    mask[:, int(EdgeType.THREATENS)] = (source == robot) & (target == robot) & not_self
    mask[:, int(EdgeType.PROXIMITY)] = (source != global_node) & (target != global_node) & not_self
    mask[:, int(EdgeType.LINE_OF_SIGHT)] = (source == robot) & (target == target_node)

    if present is not None:
        if present.ndim == 1:
            present = present.unsqueeze(0)
        present_pair = (present[:, :, None] > 0.0) & (present[:, None, :] > 0.0)
        mask &= present_pair.unsqueeze(1)
    return mask


def edge_transition_targets(current_edges: torch.Tensor, next_edges: torch.Tensor) -> torch.Tensor:
    current = current_edges > 0.5
    future = next_edges > 0.5
    targets = torch.full_like(current_edges, int(EdgeEvent.STAY), dtype=torch.long)
    targets[(~current) & future] = int(EdgeEvent.ADD)
    targets[current & (~future)] = int(EdgeEvent.DELETE)
    return targets


def duration_bucket_targets(edge_sequence: torch.Tensor) -> torch.Tensor:
    """Bucket remaining edge lifetime at each step; absent edges receive -1."""

    if edge_sequence.ndim != 5:
        raise ValueError("edge_sequence must have shape [batch, time, relation, source, target]")
    present = edge_sequence > 0.5
    batch, time, relations, nodes, _ = present.shape
    remaining = torch.zeros_like(edge_sequence, dtype=torch.long)
    run_length = torch.zeros(batch, relations, nodes, nodes, dtype=torch.long, device=edge_sequence.device)
    for step in range(time - 1, -1, -1):
        run_length = torch.where(present[:, step], run_length + 1, torch.zeros_like(run_length))
        remaining[:, step] = run_length
    buckets = torch.full_like(remaining, -1)
    buckets[remaining == 1] = 0
    buckets[(remaining >= 2) & (remaining <= 3)] = 1
    buckets[(remaining >= 4) & (remaining <= 7)] = 2
    buckets[remaining >= 8] = 3
    return buckets


class ActionConditionedConstraintGraphDynamics(nn.Module):
    """Predict typed edge creation, deletion, survival, and duration."""

    def __init__(
        self,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
        *,
        rank: int = 24,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.hidden_dim = int(hidden_dim)
        self.rank = max(int(rank), 4)
        self.node_encoder = _mlp(BELIEF_TOKEN_DIM, hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(NUM_NODE_TYPES, hidden_dim)
        projected = NUM_EDGE_TYPES * OUTPUT_HEAD_COUNT * self.rank
        self.source_projection = nn.Linear(hidden_dim, projected)
        self.target_projection = nn.Linear(hidden_dim, projected)
        self.action_bias = _mlp(
            action_dim * num_agents,
            hidden_dim,
            NUM_EDGE_TYPES * OUTPUT_HEAD_COUNT,
        )
        self.current_edge_bias = nn.Parameter(torch.zeros(NUM_EDGE_TYPES, OUTPUT_HEAD_COUNT))

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        node_types: torch.Tensor | None = None,
        current_edges: torch.Tensor | None = None,
        diagnostics: bool = True,
    ) -> dict[str, torch.Tensor]:
        if node_types is None:
            node_types = canonical_node_types_torch(tokens.shape[0], tokens.device)
        if current_edges is None:
            current_edges = build_typed_edges(tokens, node_types)
        hidden = self.node_encoder(tokens) + self.type_embedding(node_types)
        batch, nodes, _ = hidden.shape
        shape = (batch, nodes, NUM_EDGE_TYPES, OUTPUT_HEAD_COUNT, self.rank)
        source = self.source_projection(hidden).reshape(shape).permute(0, 2, 3, 1, 4)
        target = self.target_projection(hidden).reshape(shape).permute(0, 2, 3, 1, 4)
        action_bias = self.action_bias(actions.reshape(batch, -1)).reshape(
            batch, NUM_EDGE_TYPES, OUTPUT_HEAD_COUNT, 1, 1
        )
        if diagnostics:
            raw = torch.einsum("broik,brojk->broij", source, target) / math.sqrt(self.rank)
            edge_bias = current_edges.unsqueeze(2) * self.current_edge_bias[None, :, :, None, None]
            raw = raw + action_bias + edge_bias
            presence_logits = raw[:, :, 0]
        else:
            presence_logits = torch.einsum(
                "brik,brjk->brij", source[:, :, 0], target[:, :, 0]
            ) / math.sqrt(self.rank)
            presence_logits = (
                presence_logits
                + action_bias[:, :, 0]
                + current_edges * self.current_edge_bias[None, :, 0, None, None]
            )
        valid_mask = typed_edge_valid_mask(node_types, tokens[..., TOKEN_PRESENT])
        presence_logits = presence_logits.masked_fill(~valid_mask, -20.0)
        next_edge_prob = torch.sigmoid(presence_logits) * valid_mask.to(tokens.dtype)
        result = {
            "presence_logits": presence_logits,
            "next_edge_prob": next_edge_prob,
            "current_edges": current_edges,
            "valid_mask": valid_mask,
        }
        if diagnostics:
            result.update(
                {
                    "event_logits": raw[:, :, 1:1 + NUM_EDGE_EVENTS].permute(0, 1, 3, 4, 2),
                    "hazard_logits": raw[:, :, 1 + NUM_EDGE_EVENTS],
                    "duration_logits": raw[:, :, 2 + NUM_EDGE_EVENTS:].permute(0, 1, 3, 4, 2),
                }
            )
        return result

    def loss(
        self,
        prediction: dict[str, torch.Tensor],
        next_edges: torch.Tensor,
        *,
        duration_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = prediction["valid_mask"]
        target_presence = (next_edges > 0.5).to(prediction["presence_logits"].dtype)
        positives = target_presence[valid].sum()
        negatives = valid.sum().to(target_presence.dtype) - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 25.0)
        presence_loss = F.binary_cross_entropy_with_logits(
            prediction["presence_logits"][valid],
            target_presence[valid],
            pos_weight=pos_weight,
        )

        event_targets = edge_transition_targets(prediction["current_edges"], next_edges)
        event_loss = F.cross_entropy(prediction["event_logits"][valid], event_targets[valid])
        current = (prediction["current_edges"] > 0.5) & valid
        hazard_target = ((prediction["current_edges"] > 0.5) & (next_edges <= 0.5)).to(
            prediction["hazard_logits"].dtype
        )
        if current.any():
            hazard_loss = F.binary_cross_entropy_with_logits(
                prediction["hazard_logits"][current], hazard_target[current]
            )
        else:
            hazard_loss = prediction["hazard_logits"].sum() * 0.0

        duration_loss = prediction["duration_logits"].sum() * 0.0
        if duration_targets is not None:
            duration_valid = valid & (duration_targets >= 0)
            if duration_valid.any():
                duration_loss = F.cross_entropy(
                    prediction["duration_logits"][duration_valid],
                    duration_targets[duration_valid],
                )
        total = presence_loss + 0.5 * event_loss + 0.25 * hazard_loss + 0.25 * duration_loss
        return total, {
            "edge_presence_loss": presence_loss.detach(),
            "edge_event_loss": event_loss.detach(),
            "edge_hazard_loss": hazard_loss.detach(),
            "edge_duration_loss": duration_loss.detach(),
        }


def interventional_edge_consistency_loss(
    factual_probability: torch.Tensor,
    intervention_probability: torch.Tensor,
    factual_target: torch.Tensor,
    intervention_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    predicted_effect = intervention_probability - factual_probability
    target_effect = intervention_target.to(predicted_effect.dtype) - factual_target.to(predicted_effect.dtype)
    if not valid_mask.any():
        return predicted_effect.sum() * 0.0
    return F.smooth_l1_loss(predicted_effect[valid_mask], target_effect[valid_mask])
