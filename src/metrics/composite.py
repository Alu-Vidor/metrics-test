"""Composite metric helpers."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def normalize(metric_values: Iterable[float], scheme: str) -> np.ndarray:
    """Normalize metric values according to the chosen scheme."""

    values = np.asarray(list(metric_values), dtype=np.float64)
    if values.size == 0:
        return values
    if scheme == "minmax":
        vmin = np.min(values)
        vmax = np.max(values)
        if np.isclose(vmax, vmin):
            return np.ones_like(values)
        return (values - vmin) / (vmax - vmin)
    if scheme == "zscore":
        mean = np.mean(values)
        std = np.std(values) + 1e-8
        normalized = (values - mean) / std
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
        return normalized
    raise ValueError(f"Unknown normalization scheme: {scheme}")


def cost(latencies: Iterable[float], reference: float) -> np.ndarray:
    """Normalize latency/FLOPs with respect to a reference."""

    values = np.asarray(list(latencies), dtype=np.float64)
    return values / (reference + 1e-8)


def composite(m_acc: float, m_struct: float, m_rob: float, alpha: float, beta: float, gamma: float, cost: float | None = None, lam: float = 0.0) -> float:
    """Compute the weighted composite score."""

    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError("Weights must sum to 1")
    score = alpha * m_acc + beta * m_struct + gamma * m_rob
    if cost is not None and lam != 0.0:
        score -= lam * cost
    return float(score)
