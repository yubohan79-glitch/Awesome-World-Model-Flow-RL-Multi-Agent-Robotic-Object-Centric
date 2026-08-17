from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import (
    BattleConfig,
    policy_params
)

class LargeScaleBattle50v50:
    def __init__(self, config: BattleConfig | None = None):
        self.cfg = config or BattleConfig()
        self.zones = np.array([[34.0, 13.0], [40.0, 25.0], [46.0, 37.0]], dtype=np.float64)
        self.yellow_base = np.array([4.5, 25.0], dtype=np.float64)
        self.blue_base = np.array([75.5, 25.0], dtype=np.float64)
        self.obstacles = np.array(
            [
                [25.0, 6.0, 28.0, 18.5],
                [52.0, 31.5, 55.0, 44.0],
                [37.6, 21.0, 42.4, 29.0],
            ],
            dtype=np.float64,
        )

    def _initial_positions(self, team: str, rng: np.random.Generator) -> np.ndarray:
        n = self.cfg.agents_per_team
        rows = min(5, n)
        cols = int(math.ceil(n / rows))
        grid = []
        for r in range(rows):
            for c in range(cols):
                grid.append((r, c))
        grid = np.array(grid[:n], dtype=np.float64)
        y = 7.0 + grid[:, 0] * 8.5 + rng.normal(0.0, 0.25, size=n)
        if team == "yellow":
            x = 7.0 + grid[:, 1] * 0.9 + rng.normal(0.0, 0.15, size=n)
        else:
            y = self.cfg.height_m - y
            x = 73.0 - grid[:, 1] * 0.9 + rng.normal(0.0, 0.15, size=n)
        return np.stack([x, y], axis=1)

    def _obstacle_repulsion(self, pos: np.ndarray) -> tuple[np.ndarray, int]:
        force = np.zeros_like(pos)
        contacts = 0
        margin = self.cfg.obstacle_margin_m
        for rect in self.obstacles:
            xmin, ymin, xmax, ymax = rect
            closest_x = np.clip(pos[:, 0], xmin, xmax)
            closest_y = np.clip(pos[:, 1], ymin, ymax)
            diff = pos - np.stack([closest_x, closest_y], axis=1)
            dist = np.linalg.norm(diff, axis=1)
            inside = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax) & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax)
            contacts += int(np.count_nonzero(inside))
            if np.any(inside):
                left = np.abs(pos[:, 0] - xmin)
                right = np.abs(xmax - pos[:, 0])
                down = np.abs(pos[:, 1] - ymin)
                up = np.abs(ymax - pos[:, 1])
                nearest = np.stack([left, right, down, up], axis=1).argmin(axis=1)
                push = np.zeros_like(pos)
                push[nearest == 0, 0] = -1.0
                push[nearest == 1, 0] = 1.0
                push[nearest == 2, 1] = -1.0
                push[nearest == 3, 1] = 1.0
                diff[inside] = push[inside]
                dist[inside] = 0.01
            active = dist < margin
            force[active] += diff[active] / (dist[active, None] + 1e-6) * (margin - dist[active, None])
        return force, contacts

    def _separation(self, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-6
        weight = np.clip(self.cfg.separation_radius_m - dist, 0.0, None)
        weight *= alive[None, :] * alive[:, None]
        np.fill_diagonal(weight, 0.0)
        return np.sum(delta / dist[:, :, None] * weight[:, :, None], axis=1)

    def _nearest_enemy(self, own_pos: np.ndarray, own_alive: np.ndarray, enemy_pos: np.ndarray, enemy_alive: np.ndarray):
        delta = enemy_pos[None, :, :] - own_pos[:, None, :]
        dist = np.linalg.norm(delta, axis=2)
        dist = np.where(enemy_alive[None, :] & own_alive[:, None], dist, 1e9)
        idx = np.argmin(dist, axis=1)
        nearest_dist = dist[np.arange(len(own_pos)), idx]
        nearest_vec = enemy_pos[idx] - own_pos
        nearest_vec = np.where(nearest_dist[:, None] < 1e8, nearest_vec, 0.0)
        return idx, nearest_dist, nearest_vec

    def _segment_blocked(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if len(src) == 0:
            return np.zeros((0,), dtype=bool)
        t = np.linspace(0.15, 0.85, 5, dtype=np.float64)
        points = src[:, None, :] * (1.0 - t[None, :, None]) + dst[:, None, :] * t[None, :, None]
        blocked = np.zeros((len(src),), dtype=bool)
        for rect in self.obstacles:
            xmin, ymin, xmax, ymax = rect
            inside = (
                (points[:, :, 0] >= xmin)
                & (points[:, :, 0] <= xmax)
                & (points[:, :, 1] >= ymin)
                & (points[:, :, 1] <= ymax)
            )
            blocked |= inside.any(axis=1)
        return blocked

    def _zone_update(self, yellow_pos: np.ndarray, yellow_alive: np.ndarray, blue_pos: np.ndarray, blue_alive: np.ndarray, state: np.ndarray) -> np.ndarray:
        ydist = np.linalg.norm(yellow_pos[:, None, :] - self.zones[None, :, :], axis=2)
        bdist = np.linalg.norm(blue_pos[:, None, :] - self.zones[None, :, :], axis=2)
        yc = np.sum((ydist <= self.cfg.capture_radius_m) & yellow_alive[:, None], axis=0)
        bc = np.sum((bdist <= self.cfg.capture_radius_m) & blue_alive[:, None], axis=0)
        influence = (yc - bc) / (yc + bc + 4.0)
        return np.clip(state + self.cfg.capture_rate * influence, -1.0, 1.0)

    def _policy_velocity(
        self,
        team: str,
        pos: np.ndarray,
        alive: np.ndarray,
        hp: np.ndarray,
        enemy_pos: np.ndarray,
        enemy_alive: np.ndarray,
        zone_state: np.ndarray,
        shield_open: bool,
        progress_ratio: float,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        p = policy_params(theta)
        n = self.cfg.agents_per_team
        squad = np.floor(np.arange(n) * 5 / max(1, n)).astype(int)
        raw_zone_idx = squad % 3
        zone_idx = raw_zone_idx if team == "yellow" else 2 - raw_zone_idx
        flank = np.where((np.arange(n) % 2) == 0, 1.0, -1.0)
        flank_dir = flank if team == "yellow" else -flank
        side = 1.0 if team == "yellow" else -1.0
        own_base = self.yellow_base if team == "yellow" else self.blue_base
        enemy_base = self.blue_base if team == "yellow" else self.yellow_base
        zone_targets = self.zones[zone_idx].copy()
        zone_targets[:, 1] += flank_dir * (0.4 * p["spread_m"] + 0.35 * p["flank_bias_m"])
        base_targets = np.repeat(enemy_base[None, :], n, axis=0)
        base_targets[:, 0] -= side * 3.5
        base_targets[:, 1] += flank_dir * p["flank_bias_m"]

        idx, nearest_dist, nearest_vec = self._nearest_enemy(pos, alive, enemy_pos, enemy_alive)
        enemy_dir = nearest_vec / (nearest_dist[:, None] + 1e-6)
        enemy_active = nearest_dist < self.cfg.sensor_range_m

        centroids = np.zeros_like(pos)
        for s in range(5):
            mask = (squad == s) & alive
            if np.any(mask):
                centroids[squad == s] = pos[mask].mean(axis=0)
            else:
                centroids[squad == s] = own_base

        low_hp = hp < p["retreat_health"]
        controlled = np.count_nonzero(zone_state > 0.35) if team == "yellow" else np.count_nonzero(zone_state < -0.35)
        base_gate = shield_open or controlled >= 2 or progress_ratio > 0.78
        base_weight = p["base_weight"] * (6.5 if base_gate else 1.2)
        assault_mask = (squad >= 3) | (base_gate & (squad >= 2))
        defend = ((squad == 4) | low_hp) & (~assault_mask)
        defense_targets = np.repeat(own_base[None, :], n, axis=0)
        defense_targets[:, 0] += side * 8.0
        defense_targets[:, 1] += flank_dir * 6.0

        desired = np.zeros_like(pos)
        desired += p["zone_weight"] * (zone_targets - pos) * np.where(assault_mask[:, None], 0.12, 1.0)
        desired += base_weight * assault_mask[:, None] * (base_targets - pos)
        desired += p["enemy_weight"] * p["aggression"] * enemy_active[:, None] * enemy_dir * (~assault_mask)[:, None]
        desired += p["cohesion_weight"] * (centroids - pos)
        desired += p["defense_weight"] * defend[:, None] * (defense_targets - pos)
        desired += p["separation_weight"] * self._separation(pos, alive)
        obstacle_force, _ = self._obstacle_repulsion(pos)
        desired += 11.0 * obstacle_force
        desired += rng.normal(0.0, 0.08, size=desired.shape)
        desired[~alive] = 0.0

        norm = np.linalg.norm(desired, axis=1, keepdims=True)
        direction = desired / (norm + 1e-6)
        speed = self.cfg.max_speed_mps * (0.68 + 0.32 * p["aggression"])
        speed_scale = np.where((nearest_dist < self.cfg.fire_range_m * 0.9) & (~assault_mask), 0.42, 1.0)
        speed_scale = np.where(low_hp, 0.72, speed_scale)
        return direction * speed * speed_scale[:, None]

    def _apply_shots(
        self,
        shooter_team: str,
        shooter_pos: np.ndarray,
        shooter_alive: np.ndarray,
        shooter_cd: np.ndarray,
        target_pos: np.ndarray,
        target_alive: np.ndarray,
        target_hp: np.ndarray,
        base_open: bool,
        base_hp: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
        enemy_base = self.blue_base if shooter_team == "yellow" else self.yellow_base
        base_dist = np.linalg.norm(shooter_pos - enemy_base[None, :], axis=1)
        base_candidates = shooter_alive & (shooter_cd <= 0.0) & (base_dist <= self.cfg.base_fire_range_m)
        base_ids = np.flatnonzero(base_candidates)
        base_blocked = self._segment_blocked(shooter_pos[base_ids], np.repeat(enemy_base[None, :], len(base_ids), axis=0)) if len(base_ids) else np.zeros((0,), dtype=bool)
        base_legal = base_ids[~base_blocked]
        shielded = 0
        base_damage = 0.0
        if len(base_legal):
            if base_open:
                chance = 0.55 + 0.30 * (1.0 - base_dist[base_legal] / self.cfg.base_fire_range_m)
                base_hits = base_legal[rng.random(len(base_legal)) < chance]
                damage_multiplier = self.cfg.blue_base_damage_multiplier if shooter_team == "blue" else 1.0
                base_damage = float(len(base_hits) * self.cfg.base_damage * damage_multiplier)
                base_hp = max(0.0, base_hp - base_damage)
                shooter_cd[base_hits] = self.cfg.fire_cooldown_s
            else:
                shielded = int(len(base_legal))
                shooter_cd[base_legal] = self.cfg.fire_cooldown_s

        idx, nearest_dist, _ = self._nearest_enemy(shooter_pos, shooter_alive, target_pos, target_alive)
        can_fire = shooter_alive & (shooter_cd <= 0.0) & (nearest_dist <= self.cfg.fire_range_m)
        shooter_ids = np.flatnonzero(can_fire)
        blocked = self._segment_blocked(shooter_pos[shooter_ids], target_pos[idx[shooter_ids]]) if len(shooter_ids) else np.zeros((0,), dtype=bool)
        legal_ids = shooter_ids[~blocked]
        hit_chance = 0.62 + 0.26 * (1.0 - nearest_dist[legal_ids] / self.cfg.fire_range_m)
        hits = legal_ids[rng.random(len(legal_ids)) < hit_chance]
        damage = np.zeros_like(target_hp)
        if len(hits):
            np.add.at(damage, idx[hits], self.cfg.agent_damage)
            shooter_cd[hits] = self.cfg.fire_cooldown_s
        target_hp = np.maximum(0.0, target_hp - damage)

        stats = {
            "agent_shots": float(len(legal_ids)),
            "agent_hits": float(len(hits)),
            "base_shots": float(len(base_legal)),
            "base_damage": float(base_damage),
            "shielded_base_shots": float(shielded),
        }
        return shooter_cd, target_hp, base_hp, stats

    def run_episode(
        self,
        theta_yellow: np.ndarray,
        theta_blue: np.ndarray,
        seed: int,
        collect_trace: bool = False,
        trace_stride: int = 2,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        c = self.cfg
        yp = self._initial_positions("yellow", rng)
        bp = self._initial_positions("blue", rng)
        yhp = np.full(c.agents_per_team, c.agent_hp, dtype=np.float64)
        bhp = np.full(c.agents_per_team, c.agent_hp, dtype=np.float64)
        ycd = np.zeros(c.agents_per_team, dtype=np.float64)
        bcd = np.zeros(c.agents_per_team, dtype=np.float64)
        ybase = c.base_hp
        bbase = c.base_hp
        zone_state = np.zeros(3, dtype=np.float64)
        yellow_shield_progress = 0.0
        blue_shield_progress = 0.0

        stats = {
            "yellow_agent_hits": 0.0,
            "blue_agent_hits": 0.0,
            "yellow_base_damage": 0.0,
            "blue_base_damage": 0.0,
            "yellow_shielded_base_shots": 0.0,
            "blue_shielded_base_shots": 0.0,
            "robot_contacts": 0,
            "obstacle_contacts": 0,
            "yellow_zone_steps": 0,
            "blue_zone_steps": 0,
            "yellow_base_open_steps": 0,
            "blue_base_open_steps": 0,
        }
        trace = []

        for step in range(c.max_steps):
            ya = yhp > 0.0
            ba = bhp > 0.0
            if not np.any(ya) or not np.any(ba) or ybase <= 0.0 or bbase <= 0.0:
                break
            zone_state = self._zone_update(yp, ya, bp, ba, zone_state)
            yellow_control = int(np.count_nonzero(zone_state > 0.35))
            blue_control = int(np.count_nonzero(zone_state < -0.35))
            stats["yellow_zone_steps"] += yellow_control
            stats["blue_zone_steps"] += blue_control
            yellow_shield_progress = min(c.shield_progress_to_open, yellow_shield_progress + yellow_control * c.dt_s)
            blue_shield_progress = min(c.shield_progress_to_open, blue_shield_progress + blue_control * c.dt_s)
            yellow_base_open = yellow_shield_progress >= c.shield_progress_to_open
            blue_base_open = blue_shield_progress >= c.shield_progress_to_open
            stats["yellow_base_open_steps"] += int(yellow_base_open)
            stats["blue_base_open_steps"] += int(blue_base_open)

            yv = self._policy_velocity("yellow", yp, ya, yhp, bp, ba, zone_state, yellow_base_open, yellow_shield_progress / c.shield_progress_to_open, theta_yellow, rng)
            bv = self._policy_velocity("blue", bp, ba, bhp, yp, ya, zone_state, blue_base_open, blue_shield_progress / c.shield_progress_to_open, theta_blue, rng)
            yp = yp + yv * c.dt_s
            bp = bp + bv * c.dt_s
            yp[:, 0] = np.clip(yp[:, 0], 1.0, c.width_m - 1.0)
            yp[:, 1] = np.clip(yp[:, 1], 1.0, c.height_m - 1.0)
            bp[:, 0] = np.clip(bp[:, 0], 1.0, c.width_m - 1.0)
            bp[:, 1] = np.clip(bp[:, 1], 1.0, c.height_m - 1.0)
            yobs, yc = self._obstacle_repulsion(yp)
            bobs, bc = self._obstacle_repulsion(bp)
            yp += 0.95 * yobs
            bp += 0.95 * bobs
            stats["obstacle_contacts"] += yc + bc

            pair_dist = np.linalg.norm(yp[:, None, :] - bp[None, :, :], axis=2)
            contacts = (pair_dist < c.contact_radius_m) & ya[:, None] & ba[None, :]
            stats["robot_contacts"] += int(np.count_nonzero(contacts))

            ycd = np.maximum(0.0, ycd - c.dt_s)
            bcd = np.maximum(0.0, bcd - c.dt_s)
            ycd, bhp, bbase, yshot = self._apply_shots("yellow", yp, ya, ycd, bp, ba, bhp, yellow_base_open, bbase, rng)
            bcd, yhp, ybase, bshot = self._apply_shots("blue", bp, ba, bcd, yp, ya, yhp, blue_base_open, ybase, rng)
            stats["yellow_agent_hits"] += yshot["agent_hits"]
            stats["blue_agent_hits"] += bshot["agent_hits"]
            stats["yellow_base_damage"] += yshot["base_damage"]
            stats["blue_base_damage"] += bshot["base_damage"]
            stats["yellow_shielded_base_shots"] += yshot["shielded_base_shots"]
            stats["blue_shielded_base_shots"] += bshot["shielded_base_shots"]

            if collect_trace and step % trace_stride == 0:
                trace.append(
                    {
                        "step": step,
                        "yellow_pos": yp.copy(),
                        "blue_pos": bp.copy(),
                        "yellow_alive": (yhp > 0.0).copy(),
                        "blue_alive": (bhp > 0.0).copy(),
                        "zone_state": zone_state.copy(),
                        "yellow_base_hp": float(ybase),
                        "blue_base_hp": float(bbase),
                        "yellow_base_open": bool(yellow_base_open),
                        "blue_base_open": bool(blue_base_open),
                    }
                )

        ya = yhp > 0.0
        ba = bhp > 0.0
        yellow_kills = int(c.agents_per_team - np.count_nonzero(ba))
        blue_kills = int(c.agents_per_team - np.count_nonzero(ya))
        yellow_score = yellow_kills * 1.2 + (c.base_hp - bbase) * 5.0 + stats["yellow_zone_steps"] * 0.03 + stats["yellow_base_open_steps"] * 0.05 + np.count_nonzero(ya) * 0.03
        blue_score = blue_kills * 1.2 + (c.base_hp - ybase) * 5.0 + stats["blue_zone_steps"] * 0.03 + stats["blue_base_open_steps"] * 0.05 + np.count_nonzero(ba) * 0.03
        if bbase <= 0.0 and ybase > 0.0:
            winner = "yellow"
        elif ybase <= 0.0 and bbase > 0.0:
            winner = "blue"
        elif abs(yellow_score - blue_score) < 1e-6:
            winner = "draw"
        else:
            winner = "yellow" if yellow_score > blue_score else "blue"

        result: dict[str, Any] = {
            "winner": winner,
            "elapsed_s": round(step * c.dt_s, 3),
            "steps": int(step),
            "yellow_score": float(yellow_score),
            "blue_score": float(blue_score),
            "yellow_alive": int(np.count_nonzero(ya)),
            "blue_alive": int(np.count_nonzero(ba)),
            "yellow_kills": yellow_kills,
            "blue_kills": blue_kills,
            "yellow_base_hp": float(ybase),
            "blue_base_hp": float(bbase),
            "final_zone_state": zone_state.tolist(),
            "robot_contacts": int(stats["robot_contacts"]),
            "obstacle_contacts": int(stats["obstacle_contacts"]),
            "yellow_agent_hits": float(stats["yellow_agent_hits"]),
            "blue_agent_hits": float(stats["blue_agent_hits"]),
            "yellow_base_damage": float(stats["yellow_base_damage"]),
            "blue_base_damage": float(stats["blue_base_damage"]),
            "yellow_shielded_base_shots": float(stats["yellow_shielded_base_shots"]),
            "blue_shielded_base_shots": float(stats["blue_shielded_base_shots"]),
            "yellow_base_open_rate": float(stats["yellow_base_open_steps"] / max(1, step + 1)),
            "blue_base_open_rate": float(stats["blue_base_open_steps"] / max(1, step + 1)),
        }
        if collect_trace:
            result["trace"] = trace
        return result
