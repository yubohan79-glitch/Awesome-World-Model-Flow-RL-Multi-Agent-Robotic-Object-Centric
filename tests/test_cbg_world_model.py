from __future__ import annotations

import numpy as np
import torch
from evaluate_cbg_world_model import environment_counterfactual_geometry
from planning import FlowProposalRiskMPC, cvar_lower_tail, cvar_upper_tail
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from train_world_model_sacflow_selfplay import MultiAgentFlowActors
from world_model import (
    BELIEF_TOKEN_DIM,
    NUM_BELIEF_NODES,
    NUM_RULE_RISKS,
    BeliefTracker,
    CounterfactualBeliefGraphWorldModel,
    EdgeType,
    build_typed_edges,
    canonical_node_types_torch,
    extract_rule_risks,
)
from world_model.belief_graph import (
    ARMOR_BLOCKER_SLICE,
    BOX_SLICE,
    TARGET_SLICE,
    TOKEN_AGE,
    TOKEN_ATTRIBUTE_A,
    TOKEN_ATTRIBUTE_B,
    TOKEN_COVARIANCE,
    TOKEN_OCCLUDED,
    TOKEN_PRESENT,
    TOKEN_VISIBLE,
    TOKEN_X,
    TOKEN_Y,
)
from world_model.cbg_world_model import BeliefGraphDynamicsMember


def _synthetic_tokens(batch: int = 1) -> torch.Tensor:
    tokens = torch.zeros(batch, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)
    tokens[..., TOKEN_PRESENT] = 1.0
    tokens[..., TOKEN_VISIBLE] = 1.0
    tokens[:, 1, TOKEN_ATTRIBUTE_A] = 1.0
    tokens[:, 2, TOKEN_ATTRIBUTE_A] = -1.0
    tokens[:, TARGET_SLICE.start, TOKEN_ATTRIBUTE_A] = -1.0
    tokens[:, TARGET_SLICE.start, TOKEN_ATTRIBUTE_B] = 1.0
    tokens[:, ARMOR_BLOCKER_SLICE.start, TOKEN_ATTRIBUTE_A] = -1.0
    tokens[..., TOKEN_X:TOKEN_Y + 1] = torch.rand(batch, NUM_BELIEF_NODES, 2) * 1.6 - 0.8
    return tokens


def test_belief_tracker_increases_age_and_covariance_when_detection_drops():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=12)
    target = min(env.targets, key=lambda item: item.name)
    env.poses["yellow"][:2] = np.asarray(target.xy, dtype=np.float32) + np.array([0.08, 0.0], dtype=np.float32)
    tracker = BeliefTracker(observation_dropout=0.0, sensor_delay_steps=0, seed=12)

    visible = tracker.observe(env).tokens
    target_index = next(
        index
        for index in range(TARGET_SLICE.start, TARGET_SLICE.stop)
        if visible[index, TOKEN_VISIBLE] > 0.5
    )
    covariance_before = float(visible[target_index, TOKEN_COVARIANCE])

    tracker.observation_dropout = 1.0
    env.elapsed += env.dt
    occluded = tracker.observe(env).tokens

    assert occluded[target_index, TOKEN_VISIBLE] == 0.0
    assert occluded[target_index, TOKEN_OCCLUDED] == 1.0
    assert occluded[target_index, TOKEN_AGE] > 0.0
    assert occluded[target_index, TOKEN_COVARIANCE] > covariance_before


def test_no_belief_uncertainty_keeps_sensor_process_but_zeros_uncertainty_fields():
    env_enabled = RoboCupVisionRLSelfPlayEnv()
    env_disabled = RoboCupVisionRLSelfPlayEnv()
    env_enabled.reset(seed=120)
    env_disabled.reset(seed=120)
    common = dict(
        observation_dropout=0.35,
        sensor_delay_steps=2,
        covariance_growth=0.2,
        seed=991,
    )
    enabled = BeliefTracker(uncertainty_enabled=True, **common)
    disabled = BeliefTracker(uncertainty_enabled=False, **common)

    for step in range(6):
        env_enabled.elapsed += env_enabled.dt
        env_disabled.elapsed += env_disabled.dt
        enabled_tokens = enabled.observe(env_enabled).tokens.copy()
        disabled_tokens = disabled.observe(env_disabled).tokens.copy()

        retained = [
            index
            for index in range(BELIEF_TOKEN_DIM)
            if index not in (TOKEN_AGE, TOKEN_COVARIANCE, TOKEN_OCCLUDED)
        ]
        assert np.array_equal(disabled_tokens[:, retained], enabled_tokens[:, retained]), step
        assert np.count_nonzero(disabled_tokens[:, TOKEN_AGE]) == 0
        assert np.count_nonzero(disabled_tokens[:, TOKEN_COVARIANCE]) == 0
        assert np.count_nonzero(disabled_tokens[:, TOKEN_OCCLUDED]) == 0


def test_armor_presence_tracks_hit_event_even_while_geometry_is_occluded():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=13)
    tracker = BeliefTracker(sensor_delay_steps=0, seed=13)
    before = tracker.observe(env).tokens
    before_count = int((before[ARMOR_BLOCKER_SLICE, TOKEN_PRESENT] > 0.5).sum())

    env.armor["blue"] -= 1
    env.elapsed += env.dt
    after = tracker.observe(env).tokens
    after_count = int((after[ARMOR_BLOCKER_SLICE, TOKEN_PRESENT] > 0.5).sum())

    assert after_count < before_count


def test_typed_edges_are_equivariant_to_joint_node_permutation():
    tokens = _synthetic_tokens()
    node_types = canonical_node_types_torch(1, "cpu")
    edges = build_typed_edges(tokens, node_types)
    permutation = torch.randperm(NUM_BELIEF_NODES)
    permuted = build_typed_edges(tokens[:, permutation], node_types[:, permutation])

    expected = edges[:, :, permutation][:, :, :, permutation]
    assert torch.equal(permuted, expected)


def test_counterfactual_box_and_armor_interventions_change_typed_edges():
    tokens = torch.zeros(1, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)
    types = canonical_node_types_torch(1, "cpu")
    robot = 1
    target = TARGET_SLICE.start
    box = BOX_SLICE.start
    blocker = ARMOR_BLOCKER_SLICE.start
    tokens[:, [0, robot, target, box, blocker], TOKEN_PRESENT] = 1.0
    tokens[:, robot, TOKEN_ATTRIBUTE_A] = 1.0
    tokens[:, target, TOKEN_ATTRIBUTE_A] = -1.0
    tokens[:, target, TOKEN_ATTRIBUTE_B] = -1.0
    tokens[:, blocker, TOKEN_ATTRIBUTE_A] = -1.0
    tokens[:, robot, TOKEN_X:TOKEN_Y + 1] = torch.tensor([-0.8, 0.0])
    tokens[:, target, TOKEN_X:TOKEN_Y + 1] = torch.tensor([0.8, 0.0])
    tokens[:, box, TOKEN_X:TOKEN_Y + 1] = torch.tensor([0.0, 0.0])
    tokens[:, box, 8:10] = 0.12
    tokens[:, blocker, TOKEN_X:TOKEN_Y + 1] = torch.tensor([0.65, 0.0])

    factual = build_typed_edges(tokens, types)
    moved_box = tokens.clone()
    moved_box[:, box, TOKEN_Y] = 0.8
    after_push = build_typed_edges(moved_box, types)
    removed_armor = tokens.clone()
    removed_armor[:, blocker, TOKEN_PRESENT] = 0.0
    after_armor_hit = build_typed_edges(removed_armor, types)

    assert factual[0, EdgeType.BLOCKS_ROUTE, box, target] == 1.0
    assert after_push[0, EdgeType.BLOCKS_ROUTE, box, target] == 0.0
    assert factual[0, EdgeType.PROTECTS_BASE, blocker, target] == 1.0
    assert after_armor_hit[0, EdgeType.PROTECTS_BASE, blocker, target] == 0.0


def test_graph_dynamics_member_is_permutation_equivariant():
    torch.manual_seed(3)
    tokens = _synthetic_tokens(batch=2)
    actions = torch.randn(2, len(AGENTS), 6)
    types = canonical_node_types_torch(2, "cpu")
    member = BeliefGraphDynamicsMember(6, len(AGENTS), 24, graph_layers=2)
    member.eval()
    output = member(tokens, actions, types)["next_tokens"]
    permutation = torch.randperm(NUM_BELIEF_NODES)
    permuted = member(tokens[:, permutation], actions, types[:, permutation])["next_tokens"]

    assert torch.allclose(permuted, output[:, permutation], atol=1e-5, rtol=1e-5)


def test_ensemble_loss_and_multistep_rollout_are_finite():
    torch.manual_seed(4)
    tokens = _synthetic_tokens(batch=3)
    next_tokens = tokens.clone()
    next_tokens[..., :2] += 0.01
    actions = torch.randn(3, len(AGENTS), 6).clamp(-1.0, 1.0)
    rewards = torch.randn(3, len(AGENTS))
    dones = torch.zeros(3, len(AGENTS))
    risks = torch.zeros(3, len(AGENTS), NUM_RULE_RISKS)
    model = CounterfactualBeliefGraphWorldModel(6, len(AGENTS), 24, ensemble_size=2, graph_layers=1)

    loss, metrics = model.loss(tokens, actions, next_tokens, rewards, dones, risks)
    loss.backward()
    action_sequences = actions.unsqueeze(1).expand(-1, 3, -1, -1)
    rollout = model.rollout(tokens, action_sequences)

    assert torch.isfinite(loss)
    assert metrics["wm_epistemic_var"] >= 0.0
    assert rollout["tokens"].shape == (2, 3, 4, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)
    assert rollout["risk_prob"].shape == (2, 3, 3, len(AGENTS), NUM_RULE_RISKS)


def test_rollout_samples_aleatoric_particles_and_dynamic_edge_trajectories():
    torch.manual_seed(41)
    tokens = _synthetic_tokens(batch=2)
    actions = torch.zeros(2, 3, len(AGENTS), 6)
    model = CounterfactualBeliefGraphWorldModel(
        6,
        len(AGENTS),
        24,
        ensemble_size=2,
        graph_layers=1,
        learned_edge_dynamics=True,
    ).eval()
    rollout = model.rollout(tokens, actions, particles_per_member=4, sample_state=True)

    assert rollout["tokens"].shape == (8, 2, 4, NUM_BELIEF_NODES, BELIEF_TOKEN_DIM)
    assert rollout["edges"].shape[:4] == (8, 2, 4, 8)
    assert not torch.allclose(rollout["tokens"][0], rollout["tokens"][1])
    assert set(rollout["risk_cost_sample"].unique().tolist()).issubset({0.0, 1.0})


def test_sequence_and_paired_intervention_losses_are_differentiable():
    torch.manual_seed(42)
    batch, horizon = 2, 3
    tokens = _synthetic_tokens(batch).unsqueeze(1).expand(-1, horizon, -1, -1).clone()
    next_tokens = tokens.clone()
    next_tokens[..., :2] += 0.01
    actions = torch.zeros(batch, horizon, len(AGENTS), 6)
    rewards = torch.zeros(batch, horizon, len(AGENTS))
    dones = torch.zeros_like(rewards)
    risks = torch.zeros(batch, horizon, len(AGENTS), NUM_RULE_RISKS)
    model = CounterfactualBeliefGraphWorldModel(6, len(AGENTS), 16, ensemble_size=2)

    sequence_loss, sequence_metrics = model.sequence_loss(
        tokens, actions, next_tokens, rewards, dones, risks, member_indices=[1]
    )
    intervention_actions = actions.clone()
    intervention_actions[..., 0, 2] = 1.0
    intervention_next = next_tokens.clone()
    intervention_next[:, :, BOX_SLICE.start, TOKEN_Y] += 0.2
    pair_loss, pair_metrics = model.paired_intervention_loss(
        tokens,
        actions,
        next_tokens,
        rewards,
        risks,
        tokens,
        intervention_actions,
        intervention_next,
        rewards + 0.1,
        risks,
        member_indices=[1],
    )
    (sequence_loss + pair_loss).backward()

    assert torch.isfinite(sequence_loss)
    assert torch.isfinite(pair_loss)
    assert sequence_metrics["wm_sequence_edge_loss"] >= 0.0
    assert pair_metrics["wm_intervention_edge_loss"] >= 0.0
    assert all(parameter.grad is None for parameter in model.members[0].parameters())
    assert any(parameter.grad is not None for parameter in model.members[1].parameters())


def test_cvar_uses_fixed_tails_and_planner_selects_ego_only_candidates():
    values = torch.tensor([[4.0, 2.0], [-6.0, 1.0], [5.0, 0.0], [5.0, 1.0]])
    assert torch.equal(cvar_lower_tail(values, 0.25, dim=0), torch.tensor([-6.0, 0.0]))
    assert torch.equal(cvar_upper_tail(values, 0.75, dim=0), torch.tensor([5.0, 2.0]))

    torch.manual_seed(5)
    model = CounterfactualBeliefGraphWorldModel(6, len(AGENTS), 16, ensemble_size=2, graph_layers=1)
    actors = MultiAgentFlowActors(46, 6, 16, actor_mode="shared", flow_steps=1, velocity_scale=0.1)
    planner = FlowProposalRiskMPC(
        model,
        horizon=2,
        candidates=3,
        cvar_beta=0.90,
        particles_per_member=2,
        rollout_chunk_size=2,
    )
    observations = torch.zeros(1, len(AGENTS), 46)
    proposals = planner.propose(actors, observations)
    result = planner.plan(actors, observations, _synthetic_tokens())

    assert result.actions.shape == (1, len(AGENTS), 6)
    assert result.candidate_indices.shape == (1, len(AGENTS))
    assert result.scores.shape == (1, len(AGENTS), 3)
    assert result.cvar_cost.shape[-1] == NUM_RULE_RISKS
    assert torch.allclose(proposals[:, 0, :, :, 1], proposals[:, 0, :1, :, 1].expand_as(proposals[:, 0, :, :, 1]))


def test_rule_risk_labels_preserve_distinct_failure_channels():
    infos = {
        "yellow": {"robot_contact": True, "action_shield_fire": True, "line_clear": False},
        "blue": {"blocked": True, "own_target_blocked": "T01", "line_clear": True},
    }
    actions = {
        "yellow": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
        "blue": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
    }
    risks = extract_rule_risks(infos, actions)

    assert risks.shape == (len(AGENTS), NUM_RULE_RISKS)
    assert risks[0].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert risks[1].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_counterfactual_pairs_have_matching_environment_geometry_directions():
    result = environment_counterfactual_geometry(seed=2608, scenario="nominal")

    assert result["push_box"]["pair_found"] is True
    assert result["push_box"]["line_blocked_before_push"] is True
    assert result["push_box"]["line_clear_after_push"] is True
    assert result["remove_armor"]["pair_found"] is True
    assert result["remove_armor"]["line_blocked_with_full_armor"] is True
    assert result["remove_armor"]["line_clear_after_armor_hits"] is True
    assert result["remove_armor"]["armor_hits_required"] >= 1
