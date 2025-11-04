"""Statistical testing utilities."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def paired_bootstrap(scores_model_k: Iterable[float], scores_model_l: Iterable[float], n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """Paired bootstrap confidence interval of score differences."""

    a = np.asarray(list(scores_model_k), dtype=np.float64)
    b = np.asarray(list(scores_model_l), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Score arrays must have the same shape")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        diffs.append(np.mean(a[idx] - b[idx]))
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def diebold_mariano(e1: Iterable[float], e2: Iterable[float], h: int, loss: str = "L1") -> float:
    """Diebold-Mariano test statistic for equal forecast accuracy."""

    e1 = np.asarray(list(e1), dtype=np.float64)
    e2 = np.asarray(list(e2), dtype=np.float64)
    if e1.shape != e2.shape:
        raise ValueError("Error arrays must match")

    if loss.upper() == "L1":
        d_t = np.abs(e1) - np.abs(e2)
    elif loss.upper() == "L2":
        d_t = e1**2 - e2**2
    else:
        raise ValueError("Unsupported loss type")

    mean_d = np.mean(d_t)
    gamma = 0.0
    for lag in range(1, h):
        weight = 1.0 - lag / h
        gamma += weight * (np.cov(d_t[:-lag], d_t[lag:])[0, 1])
    var_d = np.var(d_t, ddof=1) + 2 * gamma
    dm_stat = mean_d / np.sqrt(var_d / len(d_t))
    return float(dm_stat)
