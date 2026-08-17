from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - mirrors the single-agent fallback.
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
        pass

    gym = _Gym()
    spaces = _Spaces()
