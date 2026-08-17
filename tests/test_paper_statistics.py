from __future__ import annotations

import numpy as np
import torch

from experiments.paper_statistics import (
    equal_mass_ece,
    fixed_tail_cvar,
    hierarchical_bootstrap_ci,
    holm_bonferroni,
    paired_permutation_pvalue,
    win_score,
)
from world_model import BELIEF_TOKEN_DIM, NUM_BELIEF_NODES, build_typed_edges, canonical_node_types_torch
from world_model.cbg_world_model import CounterfactualBeliefGraphWorldModel


def test_fixed_tail_cvar_uses_ceil_highest_cost_count():
    values = np.arange(1.0, 11.0)
    assert fixed_tail_cvar(values, beta=0.90) == 10.0
    assert fixed_tail_cvar(values, beta=0.80) == 9.5


def test_equal_mass_ece_is_zero_for_two_perfect_confidence_groups():
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    labels = probabilities.copy()
    assert equal_mass_ece(probabilities, labels, bins=2) == 0.0


def test_paired_permutation_and_holm_detect_large_registered_effect():
    strong = paired_permutation_pvalue(np.ones(12))
    weak = paired_permutation_pvalue(np.asarray([-1.0, 1.0] * 6))
    corrected = holm_bonferroni({"strong": strong, "weak": weak})
    assert corrected["strong"]["reject"] is True
    assert corrected["weak"]["reject"] is False


def test_hierarchical_bootstrap_resamples_seeds_and_world_blocks():
    values = np.asarray([0.0, 1.0, 0.5, 1.0, 0.5, 1.0])
    seed_ids = np.asarray([1, 1, 2, 2, 3, 3])
    blocks = np.asarray([10, 11, 10, 11, 10, 11])
    interval = hierarchical_bootstrap_ci(values, seed_ids, blocks, samples=200, seed=4)
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    assert interval["bootstrap_samples"] == 200


def test_win_score_treats_draw_as_half_and_is_seat_aware():
    scores = win_score(["yellow", "draw", "blue"], ["yellow", "blue", "yellow"])
    assert np.array_equal(scores, np.asarray([1.0, 0.5, 0.0]))


def test_static_rule_graph_rollout_recomputes_edges_from_each_predicted_state():
    torch.manual_seed(77)
    tokens = torch.zeros(2, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)
    tokens[..., -1] = 1.0
    actions = torch.zeros(2, 3, 2, 6)
    model = CounterfactualBeliefGraphWorldModel(
        6,
        2,
        24,
        ensemble_size=1,
        graph_layers=1,
        learned_edge_dynamics=False,
    ).eval()
    rollout = model.rollout(tokens, actions, sample_state=False)
    node_types = canonical_node_types_torch(tokens.shape[0], tokens.device)
    for step in range(actions.shape[1] + 1):
        expected = build_typed_edges(rollout["tokens"][0, :, step], node_types)
        assert torch.equal(rollout["edges"][0, :, step], expected)
