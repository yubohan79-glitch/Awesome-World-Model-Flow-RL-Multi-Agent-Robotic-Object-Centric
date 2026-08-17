from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from expert_policy import compose_policy_action
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from world_model import BeliefTracker, extract_rule_risks

from .scenario_protocol import apply_scenario


@dataclass
class InterventionBranch:
    obs: np.ndarray
    belief_state: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_belief_state: np.ndarray
    dones: np.ndarray
    rule_risks: np.ndarray
    pair_id: int
    branch: int
    exogenous_seed: int
    mechanism: str


@dataclass
class PairedInterventionTrajectory:
    factual: InterventionBranch
    intervention: InterventionBranch


def _observations_to_array(observations: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(observations[team], dtype=np.float32) for team in AGENTS])


def _actions_to_array(actions: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(actions[team], dtype=np.float32) for team in AGENTS])


def _raw_branch_actions(mechanism: str, intervention: bool) -> dict[str, np.ndarray]:
    actions = {team: np.zeros(6, dtype=np.float32) for team in AGENTS}
    for team in AGENTS:
        actions[team][4] = -1.0
    if mechanism == "push_box":
        actions["yellow"][2] = 1.0 if intervention else -1.0
        actions["yellow"][5] = 0.8 if intervention else -0.8
    elif mechanism == "remove_armor":
        actions["yellow"][4] = 1.0 if intervention else -1.0
    else:
        raise ValueError(f"unsupported intervention mechanism: {mechanism}")
    return actions


def _apply_registered_intervention(env: RoboCupVisionRLSelfPlayEnv, mechanism: str) -> None:
    if mechanism == "push_box":
        name = sorted(env.pushable_obstacles)[0]
        moved = env.pushable_obstacles[name].copy()
        moved[1] = np.clip(moved[1] + (0.42 if moved[1] <= 0.0 else -0.42), -0.96, 0.96)
        env.pushable_obstacles[name] = moved.astype(np.float32)
    elif mechanism == "remove_armor":
        env.armor["blue"] = max(0, int(env.armor["blue"]) - 1)
    else:
        raise ValueError(f"unsupported intervention mechanism: {mechanism}")


def _run_branch(
    env: RoboCupVisionRLSelfPlayEnv,
    tracker: BeliefTracker,
    initial_observations: dict[str, np.ndarray],
    *,
    mechanism: str,
    intervention: bool,
    horizon: int,
    pair_id: int,
    seed: int,
) -> InterventionBranch:
    observations = copy.deepcopy(initial_observations)
    current_belief = tracker.observe(env).flatten()
    obs_rows = []
    belief_rows = []
    action_rows = []
    reward_rows = []
    next_obs_rows = []
    next_belief_rows = []
    done_rows = []
    risk_rows = []
    for step in range(horizon):
        raw_actions = _raw_branch_actions(mechanism, intervention and step == 0)
        if intervention and step == 0:
            _apply_registered_intervention(env, mechanism)
        executed = {
            team: compose_policy_action(
                env,
                team,
                raw_actions[team],
                policy_mode="residual_expert",
                residual_scale=0.04,
            )
            for team in AGENTS
        }
        next_observations, rewards, terminations, truncations, infos = env.step(executed)
        next_belief = tracker.observe(env).flatten()
        dones = np.asarray(
            [bool(terminations[team] or truncations[team]) for team in AGENTS],
            dtype=np.float32,
        )
        obs_rows.append(_observations_to_array(observations))
        belief_rows.append(current_belief)
        action_rows.append(_actions_to_array(raw_actions))
        reward_rows.append(np.asarray([rewards[team] for team in AGENTS], dtype=np.float32))
        next_obs_rows.append(_observations_to_array(next_observations))
        next_belief_rows.append(next_belief)
        done_rows.append(dones)
        risk_rows.append(extract_rule_risks(infos, executed))
        observations = next_observations
        current_belief = next_belief
        if dones.max() > 0.5:
            break
    return InterventionBranch(
        obs=np.asarray(obs_rows, dtype=np.float32),
        belief_state=np.asarray(belief_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        next_obs=np.asarray(next_obs_rows, dtype=np.float32),
        next_belief_state=np.asarray(next_belief_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.float32),
        rule_risks=np.asarray(risk_rows, dtype=np.float32),
        pair_id=int(pair_id),
        branch=int(intervention),
        exogenous_seed=int(seed),
        mechanism=mechanism,
    )


def generate_paired_intervention(
    *,
    seed: int,
    mechanism: str,
    horizon: int = 10,
    tracker_kwargs: dict[str, object] | None = None,
    scenario: str = "nominal",
) -> PairedInterventionTrajectory:
    env = RoboCupVisionRLSelfPlayEnv(domain_randomization=True, action_shield=True)
    initial_observations, _info = env.reset(seed=seed)
    apply_scenario(env, scenario, seed)
    initial_observations = {
        team: env._obs(team) for team in AGENTS
    }
    tracker = BeliefTracker(seed=seed, **(tracker_kwargs or {}))
    environment_snapshot = copy.deepcopy(env.__dict__)
    tracker_snapshot = copy.deepcopy(tracker.__dict__)
    pair_id = int(seed * 10 + (0 if mechanism == "push_box" else 1))

    factual = _run_branch(
        env,
        tracker,
        initial_observations,
        mechanism=mechanism,
        intervention=False,
        horizon=horizon,
        pair_id=pair_id,
        seed=seed,
    )
    env.__dict__.clear()
    env.__dict__.update(copy.deepcopy(environment_snapshot))
    tracker.__dict__.clear()
    tracker.__dict__.update(copy.deepcopy(tracker_snapshot))
    intervention = _run_branch(
        env,
        tracker,
        initial_observations,
        mechanism=mechanism,
        intervention=True,
        horizon=horizon,
        pair_id=pair_id,
        seed=seed,
    )
    return PairedInterventionTrajectory(factual=factual, intervention=intervention)


def add_pair_to_replay(replay, pair: PairedInterventionTrajectory, episode_base: int) -> None:
    for branch_data in (pair.factual, pair.intervention):
        episode_id = int(episode_base + branch_data.branch)
        for step in range(branch_data.actions.shape[0]):
            replay.add(
                branch_data.obs[step],
                branch_data.belief_state[step],
                branch_data.actions[step],
                branch_data.rewards[step],
                branch_data.next_obs[step],
                branch_data.next_belief_state[step],
                branch_data.dones[step],
                branch_data.rule_risks[step],
                env_id=0,
                episode_id=episode_id,
                episode_step=step,
                pair_id=branch_data.pair_id,
                branch=branch_data.branch,
                exogenous_seed=branch_data.exogenous_seed,
            )
