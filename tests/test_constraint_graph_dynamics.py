from __future__ import annotations

import torch

from world_model import (
    BELIEF_TOKEN_DIM,
    NUM_BELIEF_NODES,
    ActionConditionedConstraintGraphDynamics,
    EdgeEvent,
    EdgeType,
    NodeType,
    canonical_node_types_torch,
    duration_bucket_targets,
    edge_transition_targets,
    typed_edge_valid_mask,
)
from world_model.belief_graph import TOKEN_PRESENT


def _tokens(batch: int = 2) -> torch.Tensor:
    tokens = torch.randn(batch, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM) * 0.05
    tokens[..., TOKEN_PRESENT] = 1.0
    return tokens


def test_typed_mask_rejects_semantically_invalid_edges():
    node_types = canonical_node_types_torch(1, "cpu")
    mask = typed_edge_valid_mask(node_types, torch.ones(1, NUM_BELIEF_NODES))
    robot = int(torch.where(node_types[0] == int(NodeType.ROBOT))[0][0])
    armor = int(torch.where(node_types[0] == int(NodeType.ARMOR_BLOCKER))[0][0])
    target = int(torch.where(node_types[0] == int(NodeType.TARGET))[0][0])

    assert mask[0, EdgeType.OBSERVES, armor, robot].item() is False
    assert mask[0, EdgeType.PROTECTS_BASE, armor, target].item() is True
    assert mask[0, EdgeType.LINE_OF_SIGHT, robot, target].item() is True


def test_edge_transition_targets_distinguish_add_delete_and_stay():
    current = torch.tensor([0.0, 1.0, 1.0, 0.0])
    future = torch.tensor([1.0, 0.0, 1.0, 0.0])
    targets = edge_transition_targets(current, future)

    assert targets.tolist() == [EdgeEvent.ADD, EdgeEvent.DELETE, EdgeEvent.STAY, EdgeEvent.STAY]


def test_duration_targets_measure_remaining_lifetime():
    edges = torch.zeros(1, 9, 1, 1, 1)
    edges[:, 0:9] = 1.0
    buckets = duration_bucket_targets(edges)

    assert buckets[0, 0, 0, 0, 0] == 3
    assert buckets[0, 2, 0, 0, 0] == 2
    assert buckets[0, 6, 0, 0, 0] == 1
    assert buckets[0, 8, 0, 0, 0] == 0


def test_constraint_graph_dynamics_is_action_conditioned_and_trainable():
    torch.manual_seed(8)
    tokens = _tokens()
    node_types = canonical_node_types_torch(tokens.shape[0], "cpu")
    model = ActionConditionedConstraintGraphDynamics(6, 2, 32, rank=8)
    actions_a = torch.zeros(tokens.shape[0], 2, 6)
    actions_b = torch.ones_like(actions_a)
    prediction_a = model(tokens, actions_a, node_types)
    prediction_b = model(tokens, actions_b, node_types)
    valid = prediction_a["valid_mask"]

    assert not torch.allclose(
        prediction_a["presence_logits"][valid], prediction_b["presence_logits"][valid]
    )
    target = prediction_a["current_edges"].clone()
    first_valid = torch.where(valid)
    target[first_valid[0][0], first_valid[1][0], first_valid[2][0], first_valid[3][0]] = 1.0
    loss, metrics = model.loss(prediction_a, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_constraint_graph_predictions_are_node_permutation_equivariant():
    torch.manual_seed(9)
    tokens = _tokens(batch=1)
    node_types = canonical_node_types_torch(1, "cpu")
    actions = torch.randn(1, 2, 6)
    model = ActionConditionedConstraintGraphDynamics(6, 2, 24, rank=8).eval()
    original = model(tokens, actions, node_types)["next_edge_prob"]
    permutation = torch.randperm(NUM_BELIEF_NODES)
    permuted = model(tokens[:, permutation], actions, node_types[:, permutation])["next_edge_prob"]
    expected = original[:, :, permutation][:, :, :, permutation]

    assert torch.allclose(permuted, expected, atol=1e-5, rtol=1e-5)
