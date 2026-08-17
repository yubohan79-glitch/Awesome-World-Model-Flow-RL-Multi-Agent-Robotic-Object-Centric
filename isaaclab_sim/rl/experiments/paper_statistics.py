from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np


def fixed_tail_cvar(values: Sequence[float] | np.ndarray, beta: float = 0.90) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("CVaR requires at least one observation")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    count = max(1, int(math.ceil((1.0 - beta) * array.size)))
    return float(np.sort(array)[-count:].mean())


def equal_mass_ece(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    bins: int = 15,
) -> float:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probability.size != target.size or probability.size == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    groups = np.array_split(np.argsort(probability, kind="stable"), min(int(bins), probability.size))
    return float(
        sum(
            group.size / probability.size
            * abs(float(probability[group].mean()) - float(target[group].mean()))
            for group in groups
            if group.size
        )
    )


def binary_brier(probabilities, labels) -> float:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.square(probability - target)))


def binary_nll(probabilities, labels) -> float:
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    target = np.asarray(labels, dtype=np.float64)
    return float(np.mean(-(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))))


def paired_permutation_pvalue(
    differences: Sequence[float] | np.ndarray,
    *,
    samples: int = 100_000,
    seed: int = 0,
) -> float:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("paired test requires finite differences")
    observed = abs(float(values.mean()))
    if values.size <= 18:
        statistics = [
            abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
            for signs in itertools.product((-1.0, 1.0), repeat=values.size)
        ]
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(int(samples), values.size))
        statistics = np.abs((signs * values[None]).mean(axis=1))
    exceedances = int(np.count_nonzero(np.asarray(statistics) >= observed - 1e-15))
    return float((exceedances + 1) / (len(statistics) + 1))


def holm_bonferroni(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(((name, float(value)) for name, value in pvalues.items()), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    still_rejecting = True
    rejected: dict[str, bool] = {}
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(running, 1.0)
        threshold = alpha / (count - rank)
        rejected[name] = bool(still_rejecting and value <= threshold)
        still_rejecting = still_rejecting and value <= threshold
    return {
        name: {
            "p_value": float(pvalues[name]),
            "adjusted_p_value": adjusted[name],
            "reject": rejected[name],
        }
        for name in pvalues
    }


def hierarchical_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    seed_ids: Sequence[int] | np.ndarray,
    block_ids: Sequence[int] | np.ndarray,
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    seeds_array = np.asarray(seed_ids).reshape(-1)
    blocks_array = np.asarray(block_ids).reshape(-1)
    if not (values_array.size == seeds_array.size == blocks_array.size) or values_array.size == 0:
        raise ValueError("values, seed_ids and block_ids must be non-empty and aligned")
    unique_seeds = np.unique(seeds_array)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(samples), dtype=np.float64)
    for sample_index in range(int(samples)):
        selected_values = []
        sampled_seeds = rng.choice(unique_seeds, size=unique_seeds.size, replace=True)
        for sampled_seed in sampled_seeds:
            seed_mask = seeds_array == sampled_seed
            seed_blocks = np.unique(blocks_array[seed_mask])
            sampled_blocks = rng.choice(seed_blocks, size=seed_blocks.size, replace=True)
            for block in sampled_blocks:
                cell = values_array[seed_mask & (blocks_array == block)]
                selected_values.append(float(rng.choice(cell)))
        estimates[sample_index] = float(np.mean(selected_values))
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": float(values_array.mean()),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "confidence": float(confidence),
        "bootstrap_samples": int(samples),
    }


def win_score(winners: Sequence[str], ego_team: Sequence[str]) -> np.ndarray:
    if len(winners) != len(ego_team):
        raise ValueError("winner and ego_team arrays must align")
    return np.asarray(
        [1.0 if winner == ego else 0.5 if winner in ("draw", "timeout") else 0.0 for winner, ego in zip(winners, ego_team)],
        dtype=np.float64,
    )
