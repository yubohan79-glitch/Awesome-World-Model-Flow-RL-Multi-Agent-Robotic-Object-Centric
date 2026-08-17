from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ShotResult:
    shooter: str
    target_name: str
    target_owner: str
    kind: str


@dataclass
class DomainRandomizationParams:
    drive_scale: float = 1.0
    turn_scale: float = 1.0
    push_step_scale: float = 1.0
    shot_accuracy_scale: float = 1.0
    drift_loss_scale: float = 1.0
    sensor_noise_scale: float = 0.0
