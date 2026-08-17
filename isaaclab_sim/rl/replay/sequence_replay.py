from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from replay_buffer import ReplayBatch


@dataclass
class SequenceBatch:
    obs: torch.Tensor
    belief_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_belief_state: torch.Tensor
    dones: torch.Tensor
    rule_risks: torch.Tensor
    episode_ids: torch.Tensor
    episode_steps: torch.Tensor
    pair_ids: torch.Tensor
    branches: torch.Tensor
    exogenous_seeds: torch.Tensor

    def first_transition(self) -> ReplayBatch:
        return ReplayBatch(
            obs=self.obs[:, 0],
            belief_state=self.belief_state[:, 0],
            actions=self.actions[:, 0],
            rewards=self.rewards[:, 0],
            next_obs=self.next_obs[:, 0],
            next_belief_state=self.next_belief_state[:, 0],
            dones=self.dones[:, 0],
            rule_risks=self.rule_risks[:, 0],
        )


@dataclass
class PairedSequenceBatch:
    factual: SequenceBatch
    intervention: SequenceBatch


class EpisodeSequenceReplay:
    """Ring replay that preserves episode order under interleaved vector writes."""

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        belief_dim: int,
        action_dim: int,
        rule_risk_dim: int,
        *,
        num_envs: int = 1,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.rule_risk_dim = int(rule_risk_dim)
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.belief_state = np.zeros((capacity, belief_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_belief_state = np.zeros((capacity, belief_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.float32)
        self.rule_risks = np.zeros((capacity, num_agents, rule_risk_dim), dtype=np.float32)
        self.episode_ids = np.full(capacity, -1, dtype=np.int64)
        self.episode_steps = np.full(capacity, -1, dtype=np.int32)
        self.pair_ids = np.full(capacity, -1, dtype=np.int64)
        self.branches = np.full(capacity, -1, dtype=np.int8)
        self.exogenous_seeds = np.full(capacity, -1, dtype=np.int64)
        self.generations = np.zeros(capacity, dtype=np.int64)
        self.index = 0
        self.size = 0
        self._next_episode_id = self.num_envs
        self._active_episode = {env_id: env_id for env_id in range(self.num_envs)}
        self._active_step = {env_id: 0 for env_id in range(self.num_envs)}
        self._episode_records: dict[int, list[tuple[int, int]]] = {}
        self._episode_pair: dict[int, tuple[int, int]] = {}

    def add(
        self,
        obs: np.ndarray,
        belief_state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        next_belief_state: np.ndarray,
        dones: np.ndarray,
        rule_risks: np.ndarray,
        *,
        env_id: int = 0,
        episode_id: int | None = None,
        episode_step: int | None = None,
        pair_id: int = -1,
        branch: int = -1,
        exogenous_seed: int = -1,
    ) -> None:
        env_id = int(env_id)
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} is outside [0, {self.num_envs})")
        resolved_episode = self._active_episode[env_id] if episode_id is None else int(episode_id)
        resolved_step = self._active_step[env_id] if episode_step is None else int(episode_step)
        slot = self.index
        self.generations[slot] += 1
        generation = int(self.generations[slot])
        self.obs[slot] = np.asarray(obs, dtype=np.float32)
        self.belief_state[slot] = np.asarray(belief_state, dtype=np.float32)
        self.actions[slot] = np.asarray(actions, dtype=np.float32)
        self.rewards[slot] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[slot] = np.asarray(next_obs, dtype=np.float32)
        self.next_belief_state[slot] = np.asarray(next_belief_state, dtype=np.float32)
        self.dones[slot] = np.asarray(dones, dtype=np.float32)
        self.rule_risks[slot] = np.asarray(rule_risks, dtype=np.float32)
        self.episode_ids[slot] = resolved_episode
        self.episode_steps[slot] = resolved_step
        self.pair_ids[slot] = int(pair_id)
        self.branches[slot] = int(branch)
        self.exogenous_seeds[slot] = int(exogenous_seed)
        self._episode_records.setdefault(resolved_episode, []).append((slot, generation))
        if pair_id >= 0 and branch >= 0:
            pair_key = (int(pair_id), int(branch))
            existing = self._episode_pair.get(resolved_episode)
            if existing is not None and existing != pair_key:
                raise ValueError("an episode cannot change pair_id or branch")
            self._episode_pair[resolved_episode] = pair_key

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        terminal = bool(np.asarray(dones).max() > 0.5)
        if episode_id is None:
            if terminal:
                self._active_episode[env_id] = self._next_episode_id
                self._next_episode_id += 1
                self._active_step[env_id] = 0
            else:
                self._active_step[env_id] = resolved_step + 1

    def _valid_episode_indices(self, episode_id: int) -> list[int]:
        records = self._episode_records.get(int(episode_id), ())
        valid = [
            slot
            for slot, generation in records
            if self.generations[slot] == generation and self.episode_ids[slot] == episode_id
        ]
        valid.sort(key=lambda slot: int(self.episode_steps[slot]))
        if not valid:
            self._episode_records.pop(int(episode_id), None)
            self._episode_pair.pop(int(episode_id), None)
            return []
        contiguous = [valid[0]]
        for slot in valid[1:]:
            if self.episode_steps[slot] == self.episode_steps[contiguous[-1]] + 1:
                contiguous.append(slot)
            else:
                contiguous = [slot]
        return contiguous

    def _sequence_candidates(self, horizon: int) -> list[tuple[int, list[int]]]:
        candidates = []
        for episode_id in list(self._episode_records):
            indices = self._valid_episode_indices(episode_id)
            if len(indices) >= horizon:
                candidates.append((episode_id, indices))
        return candidates

    def _batch_from_indices(self, indices: np.ndarray, device: torch.device) -> SequenceBatch:
        tensor = lambda value, dtype: torch.as_tensor(value[indices], dtype=dtype, device=device)
        return SequenceBatch(
            obs=tensor(self.obs, torch.float32),
            belief_state=tensor(self.belief_state, torch.float32),
            actions=tensor(self.actions, torch.float32),
            rewards=tensor(self.rewards, torch.float32),
            next_obs=tensor(self.next_obs, torch.float32),
            next_belief_state=tensor(self.next_belief_state, torch.float32),
            dones=tensor(self.dones, torch.float32),
            rule_risks=tensor(self.rule_risks, torch.float32),
            episode_ids=tensor(self.episode_ids, torch.long),
            episode_steps=tensor(self.episode_steps, torch.long),
            pair_ids=tensor(self.pair_ids, torch.long),
            branches=tensor(self.branches, torch.long),
            exogenous_seeds=tensor(self.exogenous_seeds, torch.long),
        )

    def sample_sequences(self, batch_size: int, horizon: int, device: torch.device) -> SequenceBatch:
        horizon = max(int(horizon), 1)
        candidates = self._sequence_candidates(horizon)
        if not candidates:
            raise RuntimeError(f"no contiguous replay sequences with horizon {horizon}")
        selections = []
        for _ in range(int(batch_size)):
            _episode_id, indices = candidates[int(self.rng.integers(0, len(candidates)))]
            start = int(self.rng.integers(0, len(indices) - horizon + 1))
            selections.append(indices[start:start + horizon])
        return self._batch_from_indices(np.asarray(selections, dtype=np.int64), device)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        return self.sample_sequences(batch_size, 1, device).first_transition()

    def sample_paired_sequences(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device,
    ) -> PairedSequenceBatch:
        paired: dict[int, dict[int, list[int]]] = {}
        for episode_id, pair_key in list(self._episode_pair.items()):
            pair_id, branch = pair_key
            indices = self._valid_episode_indices(episode_id)
            if len(indices) >= horizon:
                paired.setdefault(pair_id, {})[branch] = indices
        eligible = [pair_id for pair_id, branches in paired.items() if 0 in branches and 1 in branches]
        if not eligible:
            raise RuntimeError(f"no paired replay sequences with horizon {horizon}")

        factual_indices = []
        intervention_indices = []
        for _ in range(int(batch_size)):
            pair_id = eligible[int(self.rng.integers(0, len(eligible)))]
            factual = paired[pair_id][0]
            intervention = paired[pair_id][1]
            maximum = min(len(factual), len(intervention)) - horizon
            start = int(self.rng.integers(0, maximum + 1))
            factual_indices.append(factual[start:start + horizon])
            intervention_indices.append(intervention[start:start + horizon])
        return PairedSequenceBatch(
            factual=self._batch_from_indices(np.asarray(factual_indices, dtype=np.int64), device),
            intervention=self._batch_from_indices(np.asarray(intervention_indices, dtype=np.int64), device),
        )

    def __len__(self) -> int:
        return self.size
