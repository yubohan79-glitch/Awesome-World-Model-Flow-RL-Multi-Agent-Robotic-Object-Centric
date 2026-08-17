from __future__ import annotations

import numpy as np

from robocup_visionrl_selfplay_env import DomainRandomizationParams, RoboCupVisionRLSelfPlayEnv


SCENARIOS = (
    "nominal",
    "held_out_boxes",
    "held_out_target_yaw",
    "delayed_occlusion",
    "low_traction",
    "aggressive_opponent",
)


def apply_scenario(env: RoboCupVisionRLSelfPlayEnv, scenario: str, seed: int) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown registered scenario: {scenario}")
    if scenario == "held_out_boxes":
        offsets = (
            np.asarray([0.17, -0.13], dtype=np.float32),
            np.asarray([-0.16, 0.14], dtype=np.float32),
        )
        for index, name in enumerate(sorted(env.pushable_obstacles)):
            env.pushable_obstacles[name] = np.clip(
                env.pushable_obstacles[name] + offsets[index % len(offsets)], -1.0, 1.0
            ).astype(np.float32)
        env._path_cache.clear()
        env._fire_pose_cache.clear()
    elif scenario == "held_out_target_yaw":
        sign = 1.0 if int(seed) % 2 == 0 else -1.0
        for target in env.targets:
            if target.kind == "normal":
                target.yaw = float(target.yaw + sign * np.deg2rad(25.7))
        env._fire_pose_cache.clear()
    elif scenario == "low_traction":
        env.domain_params = DomainRandomizationParams(
            drive_scale=1.09,
            turn_scale=1.12,
            push_step_scale=1.25,
            shot_accuracy_scale=0.88,
            drift_loss_scale=1.55,
            sensor_noise_scale=0.04,
        )


def tracker_overrides(scenario: str) -> dict[str, object]:
    if scenario == "delayed_occlusion":
        return {
            "sensor_delay_steps": 4,
            "observation_dropout": 0.25,
            "covariance_growth": 0.14,
        }
    return {}


def aggressive_action() -> np.ndarray:
    return np.asarray([0.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
