"""Tests for metric helpers."""
from __future__ import annotations

import numpy as np

from src.metrics.accuracy import lp_error
from src.metrics.composite import composite
from src.metrics.robustness import stability_score


def test_robustness_decreases_with_noise() -> None:
    y_true = np.ones((32, 12))
    y_hat_clean = y_true.copy()
    sigmas = [0.0, 0.05, 0.1, 0.2]
    averaged_scores = []
    for sigma in sigmas:
        scores = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            noise = rng.normal(scale=sigma, size=y_true.shape)
            noisy = y_hat_clean + noise
            scores.append(stability_score(y_hat_clean, noisy, y_true, lp_error))
        averaged_scores.append(float(np.mean(scores)))

    for earlier, later in zip(averaged_scores, averaged_scores[1:]):
        assert earlier >= later - 1e-6


def test_composite_monotonic_components() -> None:
    base = composite(0.5, 0.5, 0.5, alpha=1 / 3, beta=1 / 3, gamma=1 / 3)
    improved = composite(0.6, 0.6, 0.6, alpha=1 / 3, beta=1 / 3, gamma=1 / 3)
    assert improved > base


def test_composite_penalises_cost() -> None:
    base = composite(0.6, 0.6, 0.6, alpha=1 / 3, beta=1 / 3, gamma=1 / 3, cost=0.1, lam=0.5)
    slower = composite(0.6, 0.6, 0.6, alpha=1 / 3, beta=1 / 3, gamma=1 / 3, cost=0.2, lam=0.5)
    assert slower < base
