from __future__ import annotations

import numpy as np
import torch

from experiments.paired_interventions import generate_paired_intervention
from replay import EpisodeSequenceReplay


def _add_step(
    replay: EpisodeSequenceReplay,
    *,
    env_id: int,
    value: float,
    done: bool,
    episode_id: int | None = None,
    pair_id: int = -1,
    branch: int = -1,
    seed: int = -1,
) -> None:
    obs = np.full((2, 3), value, dtype=np.float32)
    belief = np.full(5, value, dtype=np.float32)
    actions = np.full((2, 2), value, dtype=np.float32)
    rewards = np.full(2, value, dtype=np.float32)
    dones = np.full(2, float(done), dtype=np.float32)
    risks = np.zeros((2, 4), dtype=np.float32)
    replay.add(
        obs,
        belief,
        actions,
        rewards,
        obs + 1.0,
        belief + 1.0,
        dones,
        risks,
        env_id=env_id,
        episode_id=episode_id,
        episode_step=int(value),
        pair_id=pair_id,
        branch=branch,
        exogenous_seed=seed,
    )


def test_sequence_replay_preserves_contiguity_for_interleaved_envs():
    replay = EpisodeSequenceReplay(64, 2, 3, 5, 2, 4, num_envs=2, seed=3)
    for step in range(6):
        _add_step(replay, env_id=0, value=float(step), done=step == 5)
        _add_step(replay, env_id=1, value=float(step), done=step == 5)
    batch = replay.sample_sequences(8, 4, torch.device("cpu"))

    assert batch.actions.shape == (8, 4, 2, 2)
    assert torch.all(torch.diff(batch.episode_steps, dim=1) == 1)
    assert torch.all(batch.episode_ids == batch.episode_ids[:, :1])


def test_paired_sequence_sampling_keeps_branch_and_exogenous_seed_aligned():
    replay = EpisodeSequenceReplay(64, 2, 3, 5, 2, 4, num_envs=1, seed=4)
    for branch in (0, 1):
        for step in range(5):
            _add_step(
                replay,
                env_id=0,
                value=float(step + 10 * branch),
                done=step == 4,
                episode_id=100 + branch,
                pair_id=77,
                branch=branch,
                seed=9001,
            )
    paired = replay.sample_paired_sequences(3, 3, torch.device("cpu"))

    assert torch.all(paired.factual.pair_ids == 77)
    assert torch.all(paired.intervention.pair_ids == 77)
    assert torch.all(paired.factual.branches == 0)
    assert torch.all(paired.intervention.branches == 1)
    assert torch.all(paired.factual.exogenous_seeds == paired.intervention.exogenous_seeds)
    assert not torch.allclose(paired.factual.actions, paired.intervention.actions)


def test_simulator_pairs_share_initial_state_and_change_only_registered_branch():
    for mechanism in ("push_box", "remove_armor"):
        pair = generate_paired_intervention(seed=6123, mechanism=mechanism, horizon=3)

        assert pair.factual.pair_id == pair.intervention.pair_id
        assert pair.factual.exogenous_seed == pair.intervention.exogenous_seed
        assert pair.factual.branch == 0
        assert pair.intervention.branch == 1
        assert np.allclose(pair.factual.belief_state[0], pair.intervention.belief_state[0])
        assert not np.allclose(pair.factual.actions[0], pair.intervention.actions[0])
        assert not np.allclose(
            pair.factual.next_belief_state[0], pair.intervention.next_belief_state[0]
        )
