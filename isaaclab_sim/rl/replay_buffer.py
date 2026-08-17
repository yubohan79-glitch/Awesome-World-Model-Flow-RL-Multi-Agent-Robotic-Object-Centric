from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    belief_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    next_belief_state: torch.Tensor
    dones: torch.Tensor
    rule_risks: torch.Tensor


class MultiAgentReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        belief_dim: int,
        action_dim: int,
        rule_risk_dim: int,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.rule_risk_dim = int(rule_risk_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.belief_state = np.zeros((capacity, belief_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_belief_state = np.zeros((capacity, belief_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.float32)
        self.rule_risks = np.zeros((capacity, num_agents, rule_risk_dim), dtype=np.float32)
        self.index = 0
        self.size = 0

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
    ) -> None:
        slot = self.index
        self.obs[slot] = np.asarray(obs, dtype=np.float32)
        self.belief_state[slot] = np.asarray(belief_state, dtype=np.float32)
        self.actions[slot] = np.asarray(actions, dtype=np.float32)
        self.rewards[slot] = np.asarray(rewards, dtype=np.float32)
        self.next_obs[slot] = np.asarray(next_obs, dtype=np.float32)
        self.next_belief_state[slot] = np.asarray(next_belief_state, dtype=np.float32)
        self.dones[slot] = np.asarray(dones, dtype=np.float32)
        self.rule_risks[slot] = np.asarray(rule_risks, dtype=np.float32)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size <= 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(
            obs=torch.as_tensor(self.obs[indices], dtype=torch.float32, device=device),
            belief_state=torch.as_tensor(self.belief_state[indices], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            next_belief_state=torch.as_tensor(self.next_belief_state[indices], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            rule_risks=torch.as_tensor(self.rule_risks[indices], dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return self.size
