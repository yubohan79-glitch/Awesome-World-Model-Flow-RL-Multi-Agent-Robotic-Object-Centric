from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Target:
    name: str
    xy: tuple[float, float]
    yaw: float
    kind: str
    owner: str
    knocked: bool = False
