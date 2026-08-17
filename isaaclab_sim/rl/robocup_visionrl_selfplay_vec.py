from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from expert_policy import compose_policy_action
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


def _compose_one(args):
    env, raw_actions, policy_mode, residual_scale = args
    return {
        team: compose_policy_action(
            env,
            team,
            raw_actions[index],
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
        for index, team in enumerate(AGENTS)
    }


def _step_one(args):
    env, actions = args
    return env.step(actions)


class RoboCupVisionRLSelfPlayVector:
    """Simple vectorized self-play runner for SAC Flow rollout collection.

    It keeps environments in-process so debugging is easy and reproducible
    while the object-centric replay buffer collects self-play transitions.
    """

    def __init__(
        self,
        num_envs: int = 16,
        seed: int = 7,
        env_kwargs: dict | None = None,
        workers: int = 0,
    ):
        kwargs = dict(env_kwargs or {})
        self.envs = [RoboCupVisionRLSelfPlayEnv(**kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed
        self.env_kwargs = kwargs
        self.workers = max(0, min(int(workers), num_envs))
        self._executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def reset(self):
        observations = []
        infos = []
        for index, env in enumerate(self.envs):
            obs, info = env.reset(seed=self.seed + index)
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def reset_one(self, index: int, seed: int | None = None):
        return self.envs[index].reset(seed=self.seed + index if seed is None else seed)

    def step(self, actions: list[dict[str, np.ndarray]]):
        items = list(zip(self.envs, actions))
        outputs = list(self._executor.map(_step_one, items)) if self._executor is not None else [_step_one(item) for item in items]
        observations, rewards, terminations, truncations, infos = zip(*outputs)
        return list(observations), list(rewards), list(terminations), list(truncations), list(infos)

    def compose_actions(
        self,
        raw_actions: np.ndarray,
        *,
        policy_mode: str,
        residual_scale: float,
    ) -> list[dict[str, np.ndarray]]:
        items = [
            (env, raw_actions[index], policy_mode, residual_scale)
            for index, env in enumerate(self.envs)
        ]
        return list(self._executor.map(_compose_one, items)) if self._executor is not None else [_compose_one(item) for item in items]

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


if __name__ == "__main__":
    vec = RoboCupVisionRLSelfPlayVector(num_envs=8)
    observations, _ = vec.reset()
    for _ in range(8):
        actions = [
            {team: vec.envs[index].action_spaces[team].sample() for team in AGENTS}
            for index in range(vec.num_envs)
        ]
        observations, rewards, terminations, truncations, infos = vec.step(actions)
        mean_yellow = sum(item["yellow"] for item in rewards) / vec.num_envs
        mean_blue = sum(item["blue"] for item in rewards) / vec.num_envs
        print(f"mean_reward yellow={mean_yellow:.3f} blue={mean_blue:.3f}")
