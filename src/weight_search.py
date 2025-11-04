"""Search utilities for component weights."""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np

from .metrics.composite import composite


def _weight_grid(step: float) -> Iterable[Tuple[float, float, float]]:
    values = np.arange(0.0, 1.0 + step / 2.0, step)
    for alpha in values:
        for beta in values:
            gamma = 1.0 - alpha - beta
            if gamma < -1e-8:
                continue
            gamma = max(0.0, gamma)
            total = alpha + beta + gamma
            if total == 0:
                continue
            yield alpha / total, beta / total, gamma / total


def find_weights(val_tables: Mapping[str, Mapping[str, Mapping[str, float]]], scenario_weights: Mapping[str, float], lam: float, grid_step: float = 0.05) -> Tuple[float, float, float]:
    """Grid-search the optimal (alpha, beta, gamma) triplet."""

    if not val_tables:
        return 1 / 3, 1 / 3, 1 / 3

    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_score = -np.inf
    for alpha, beta, gamma in _weight_grid(grid_step):
        total = 0.0
        count = 0
        for model_tables in val_tables.values():
            for scenario, metrics in model_tables.items():
                weight = scenario_weights.get(scenario, 1.0)
                score = composite(
                    metrics["accuracy"],
                    metrics["structure"],
                    metrics["robustness"],
                    alpha,
                    beta,
                    gamma,
                    cost=metrics.get("cost"),
                    lam=lam,
                )
                total += weight * score
                count += 1
        mean_score = total / max(count, 1)
        if mean_score > best_score:
            best_score = mean_score
            best_weights = (alpha, beta, gamma)
    return best_weights
