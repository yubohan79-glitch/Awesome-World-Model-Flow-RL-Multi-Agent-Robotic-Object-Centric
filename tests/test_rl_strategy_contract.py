from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from expert_policy import blue_expert_action, expert_profile, select_target, yellow_expert_action
from robocup_visionrl_selfplay_env import (
    AGENTS,
    TACTICAL_ACTION_DIM,
    TACTICAL_ACTION_LABELS,
    YELLOW_BASE_XY,
    RoboCupVisionRLSelfPlayEnv,
)


ROOT = Path(__file__).resolve().parents[1]


def test_world_model_flow_config_matches_environment_action_contract():
    config = yaml.safe_load((ROOT / "isaaclab_sim/rl/configs/world_model_flow.yaml").read_text(encoding="utf-8"))

    assert config["algorithm"] == "cbg_wm_sac_flow_selfplay"
    assert config["actor_mode"] == "dual"
    assert config["policy_mode"] == "residual_expert"
    assert config["research_stack"]["uncertainty_aware_belief_tokens"] is True
    assert config["research_stack"]["typed_interaction_graph"] is True
    assert config["research_stack"]["probabilistic_ensemble"] is True
    assert config["research_stack"]["flow_proposal_cvar_mpc"] is True
    assert config["research_stack"]["centralized_twin_q"] is True
    assert config["research_stack"]["sac_flow_actor"] is True
    assert config["research_stack"]["shielded_execution"] is True
    assert config["deployment"]["action_contract"] == list(TACTICAL_ACTION_LABELS)
    assert TACTICAL_ACTION_DIM == len(config["deployment"]["action_contract"])

    env = RoboCupVisionRLSelfPlayEnv()
    for team in AGENTS:
        assert env.action_spaces[team].shape == (TACTICAL_ACTION_DIM,)
        assert env.observation_spaces[team].shape == (46,)


def test_yellow_and_blue_experts_have_distinct_tempo_profiles():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=16)

    yellow_profile = expert_profile("yellow")
    blue_profile = expert_profile("blue")

    assert yellow_profile.name == "yellow_expert"
    assert blue_profile.name == "blue_expert"
    assert yellow_profile.normal_order != blue_profile.normal_order
    assert yellow_profile.side_gate_targets != blue_profile.side_gate_targets
    assert yellow_profile.base_risk != blue_profile.base_risk

    yellow_target = select_target(env, "yellow", risk=0.72, profile=yellow_profile)
    blue_target = select_target(env, "blue", risk=0.72, profile=blue_profile)
    assert yellow_target is not None
    assert blue_target is not None
    assert yellow_target.name in yellow_profile.normal_order
    assert blue_target.name in blue_profile.normal_order

    yellow_action = yellow_expert_action(env)
    blue_action = blue_expert_action(env)
    assert yellow_action.shape == (TACTICAL_ACTION_DIM,)
    assert blue_action.shape == (TACTICAL_ACTION_DIM,)
    assert not np.allclose(yellow_action, blue_action)


def test_tactical_target_selection_never_selects_own_side_target():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=21)

    candidate_actions = [
        np.array([-1.0, -1.0, -1.0, -1.0, 0.5, -0.5], dtype=np.float32),
        np.array([0.0, 0.8, -1.0, -1.0, 1.0, 0.2], dtype=np.float32),
        np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float32),
    ]
    for team in AGENTS:
        for action in candidate_actions:
            target = env._select_tactical_target(team, action)
            assert target is not None
            assert target.owner != team


def test_yellow_expert_allows_two_hit_base_rush_when_window_is_good():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=32)
    env.strategy_counts["yellow"]["normal_hits"] = 2
    env.armor["blue"] = 2

    target = select_target(env, "yellow", risk=0.78, profile=expert_profile("yellow"))

    assert target is not None
    assert target.owner == "blue"
    assert target.kind == "base_blue"


def test_blue_expert_allows_two_hit_base_rush_when_window_is_good():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=33)
    env.strategy_counts["blue"]["normal_hits"] = 2
    env.armor["yellow"] = 2

    target = select_target(env, "blue", risk=0.78, profile=expert_profile("blue"))

    assert target is not None
    assert target.owner == "yellow"
    assert target.kind == "base_yellow"


def test_recovery_action_enters_relocalization_branch():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=34)
    env.localization_confidence["yellow"] = 0.20

    actions = {
        "yellow": np.array([0.0, -1.0, -1.0, 1.0, -1.0, 0.0], dtype=np.float32),
        "blue": np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32),
    }
    _obs, rewards, _terminations, _truncations, infos = env.step(actions)

    assert infos["yellow"]["tactic"] == "recover"
    assert infos["yellow"]["relocalizing"] is True
    assert env.localization_confidence["yellow"] > 0.20
    assert rewards["yellow"] > -0.01


def test_recovery_action_has_cooldown_to_avoid_spin_loops():
    env = RoboCupVisionRLSelfPlayEnv()
    env.reset(seed=35)
    action = np.array([0.0, -1.0, -1.0, 1.0, -1.0, 0.0], dtype=np.float32)

    env.localization_confidence["blue"] = 0.20
    env.step({"yellow": np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32), "blue": action})
    first_relocalization = env.last_relocalization_time["blue"]
    assert first_relocalization > -1.0

    env.localization_confidence["blue"] = 0.20
    _obs, _rewards, _terminations, _truncations, infos = env.step(
        {"yellow": np.zeros(TACTICAL_ACTION_DIM, dtype=np.float32), "blue": action}
    )

    assert infos["blue"].get("relocalizing") is not True
    assert infos["blue"]["relocalization_cooldown"] is True
    assert env.last_relocalization_time["blue"] == first_relocalization


def test_domain_randomization_is_opt_in_and_bounded():
    nominal = RoboCupVisionRLSelfPlayEnv()
    nominal.reset(seed=41)
    assert nominal.domain_params.drive_scale == 1.0
    assert nominal.domain_params.shot_accuracy_scale == 1.0

    randomized = RoboCupVisionRLSelfPlayEnv(domain_randomization=True)
    randomized.reset(seed=41)
    params = randomized.domain_params
    assert 0.92 <= params.drive_scale <= 1.08
    assert 0.90 <= params.turn_scale <= 1.10
    assert 0.72 <= params.push_step_scale <= 1.18
    assert 0.82 <= params.shot_accuracy_scale <= 1.05
    assert 0.85 <= params.drift_loss_scale <= 1.40
    assert 0.0 <= params.sensor_noise_scale <= 0.035


def test_action_shield_suppresses_contact_near_own_assets():
    env = RoboCupVisionRLSelfPlayEnv(action_shield=True)
    env.reset(seed=43)
    env.poses["yellow"][:2] = np.asarray(YELLOW_BASE_XY, dtype=np.float32)
    risky = np.array([0.0, -1.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    shielded, changed = env._shield_contact_action("yellow", risky)

    assert changed is True
    assert shielded[2] <= -0.25
    assert float(shielded[5]) <= 0.20 + 1e-6


def test_published_world_model_flow_run_data_has_expected_strategy_metrics():
    summary = json.loads(
        (ROOT / "docs/rl_data/world_model_sacflow_final/training_summary.json").read_text(encoding="utf-8")
    )
    stochastic = json.loads(
        (ROOT / "docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json").read_text(encoding="utf-8")
    )

    assert summary["algorithm"] == "object_centric_world_model_sac_flow_selfplay"
    assert summary["device"] == "cuda"
    assert summary["action_dim"] == TACTICAL_ACTION_DIM
    assert summary["obs_dim"] == 46
    assert summary["object_state_dim"] > summary["obs_dim"]
    assert summary["config"]["actor_mode"] == "dual"
    assert summary["config"]["policy_mode"] == "residual_expert"

    eval_summary = stochastic["summary"]
    assert eval_summary["episodes"] >= 128
    assert eval_summary["static_penetrations_total"] == 0
    assert eval_summary["box_penetrations_total"] == 0
    assert eval_summary["repeat_target_order_events_total"] == 0
    assert 0.40 <= eval_summary["yellow_win_rate"] <= 0.60
    assert 0.40 <= eval_summary["blue_win_rate"] <= 0.60
