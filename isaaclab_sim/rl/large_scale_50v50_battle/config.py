from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np



ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs" / "rl_data" / "large_scale_50v50"
FIG_DIR = ROOT / "docs" / "figures" / "large_scale_50v50"
MEDIA_DIR = ROOT / "docs" / "media"


@dataclass
class BattleConfig:
    width_m: float = 80.0
    height_m: float = 50.0
    agents_per_team: int = 50
    dt_s: float = 0.20
    max_steps: int = 420
    max_speed_mps: float = 3.0
    fire_range_m: float = 6.5
    base_fire_range_m: float = 10.0
    fire_cooldown_s: float = 1.20
    agent_hp: float = 3.0
    agent_damage: float = 0.16
    base_hp: float = 45.0
    base_damage: float = 1.10
    blue_base_damage_multiplier: float = 1.0
    capture_radius_m: float = 6.0
    capture_rate: float = 0.055
    shield_progress_to_open: float = 9.0
    obstacle_margin_m: float = 1.1
    contact_radius_m: float = 0.70
    separation_radius_m: float = 1.35
    sensor_range_m: float = 14.0


DEFAULT_THETA = np.array(
    [2.0, 5.0, -1.0, -1.0, 2.5, 0.0, -2.0, 2.0, 1.0, -2.0],
    dtype=np.float64,
)


def config_from_args(args: argparse.Namespace) -> BattleConfig:
    cfg = BattleConfig()
    if hasattr(args, "agents_per_team"):
        cfg.agents_per_team = int(args.agents_per_team)
    if hasattr(args, "max_steps"):
        cfg.max_steps = int(args.max_steps)
    if hasattr(args, "base_hp") and args.base_hp is not None:
        cfg.base_hp = float(args.base_hp)
    if hasattr(args, "base_damage") and args.base_damage is not None:
        cfg.base_damage = float(args.base_damage)
    if hasattr(args, "blue_base_damage_multiplier") and args.blue_base_damage_multiplier is not None:
        cfg.blue_base_damage_multiplier = float(args.blue_base_damage_multiplier)
    if hasattr(args, "capture_rate") and args.capture_rate is not None:
        cfg.capture_rate = float(args.capture_rate)
    if hasattr(args, "shield_progress_to_open") and args.shield_progress_to_open is not None:
        cfg.shield_progress_to_open = float(args.shield_progress_to_open)
    if hasattr(args, "contact_radius") and args.contact_radius is not None:
        cfg.contact_radius_m = float(args.contact_radius)
    if hasattr(args, "separation_radius") and args.separation_radius is not None:
        cfg.separation_radius_m = float(args.separation_radius)
    return cfg


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def policy_params(theta: np.ndarray) -> dict[str, float]:
    theta = np.asarray(theta, dtype=np.float64)
    return {
        "zone_weight": 0.8 + 1.7 * float(sigmoid(theta[0])),
        "base_weight": 0.4 + 2.2 * float(sigmoid(theta[1])),
        "enemy_weight": 0.2 + 1.6 * float(sigmoid(theta[2])),
        "cohesion_weight": 0.1 + 1.1 * float(sigmoid(theta[3])),
        "separation_weight": 0.8 + 2.0 * float(sigmoid(theta[4])),
        "flank_bias_m": 9.0 * float(np.tanh(theta[5])),
        "defense_weight": 0.2 + 1.8 * float(sigmoid(theta[6])),
        "aggression": float(sigmoid(theta[7])),
        "spread_m": 1.0 + 4.0 * float(sigmoid(theta[8])),
        "retreat_health": 0.12 + 0.55 * float(sigmoid(theta[9])),
    }
