from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - keeps rule-env smoke tests dependency-light.
    class _Env:
        def reset(self, seed: int | None = None):
            self.np_random = np.random.default_rng(seed)

    class _Box:
        def __init__(self, low, high, dtype=None):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.dtype = dtype
            self.shape = self.low.shape

        def sample(self):
            return np.zeros(self.shape, dtype=self.dtype or np.float32)

    class _Spaces:
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()
