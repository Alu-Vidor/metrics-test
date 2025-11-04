"""Accuracy metrics normalized to [0, 1]."""
from __future__ import annotations

from typing import Optional

import numpy as np


_EPS = 1e-8


def lp_error(y_hat: np.ndarray, y: np.ndarray, p: int = 1, w_h: Optional[np.ndarray] = None) -> float:
    """Normalized Lp error score where 1 denotes a perfect forecast."""

    error = np.abs(y_hat - y) ** p
    if w_h is not None:
        if w_h.shape != y.shape:
            raise ValueError("Weight tensor must match target shape")
        error = error * w_h
    error_sum = float(np.sum(error))
    target_sum = float(np.sum(np.abs(y) ** p))
    denom = error_sum + target_sum + _EPS
    return 1.0 - error_sum / denom


def smape(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Symmetric MAPE normalized into [0, 1]."""

    numerator = np.abs(y_hat - y)
    denominator = (np.abs(y_hat) + np.abs(y)) / 2.0
    raw = np.where(denominator > 0, numerator / (denominator + _EPS), 0.0)
    raw = np.clip(raw, 0.0, None)
    return 1.0 - float(np.mean(raw))


def mase(y_hat: np.ndarray, y: np.ndarray, y_naive: np.ndarray) -> float:
    """Mean absolute scaled error transformed into a score."""

    mae_model = float(np.mean(np.abs(y - y_hat)))
    mae_naive = float(np.mean(np.abs(y - y_naive))) + _EPS
    ratio = mae_model / mae_naive
    return float(1.0 / (1.0 + ratio))
