"""Probabilistic scoring rules."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def pinball(y: np.ndarray, q_hat: np.ndarray, tau: float) -> float:
    """Pinball score converted to a [0, 1] accuracy-style metric."""

    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    diff = y - q_hat
    loss = np.maximum(tau * diff, (tau - 1) * diff)
    scale = np.mean(np.abs(y)) + 1e-8
    score = 1.0 - float(np.mean(loss) / (scale + 1e-8))
    return score


def crps(y: np.ndarray, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score approximated from samples."""

    y = np.asarray(y)
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    diff_obs = np.mean(np.abs(samples - y), axis=0)
    diff_samples = np.mean(np.abs(samples[:, None, :] - samples[None, :, :]), axis=(0, 1))
    crps_val = diff_obs - 0.5 * diff_samples
    scale = np.mean(np.abs(y)) + 1e-8
    return float(1.0 - np.mean(crps_val) / (scale + 1e-8))


def calibration_error(intervals: Sequence[tuple], y: np.ndarray) -> float:
    """Expected calibration error for prediction intervals."""

    if len(intervals) == 0:
        return 1.0
    errors = []
    for lower, upper, nominal in intervals:
        covered = np.logical_and(y >= lower, y <= upper)
        empirical = np.mean(covered)
        errors.append(abs(empirical - nominal))
    return 1.0 - float(np.mean(errors))
