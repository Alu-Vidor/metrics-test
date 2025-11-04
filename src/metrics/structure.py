"""Structure-aware metrics."""
from __future__ import annotations

import math
from typing import Optional
import numpy as np


_EPS = 1e-8


def dtw_distance(a: np.ndarray, b: np.ndarray, window: Optional[int] = None) -> float:
    """Dynamic time warping distance between two sequences."""

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    len_a, len_b = a.shape[-2], b.shape[-2]
    window = len_a if window is None else max(window, abs(len_a - len_b))

    cost = np.full((len_a + 1, len_b + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len_a + 1):
        start = max(1, i - window)
        end = min(len_b, i + window)
        for j in range(start, end + 1):
            dist = np.linalg.norm(a[..., i - 1, :] - b[..., j - 1, :])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[len_a, len_b])


def soft_dtw(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """Soft-DTW using the log-sum-exp smoothing from Cuturi & Blondel (2017)."""

    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    len_a, len_b = a.shape[-2], b.shape[-2]
    D = np.zeros((len_a, len_b))
    for i in range(len_a):
        for j in range(len_b):
            D[i, j] = np.linalg.norm(a[..., i, :] - b[..., j, :]) ** 2

    R = np.full((len_a + 1, len_b + 1), np.inf)
    R[0, 0] = 0.0
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(r0, r1, r2)
            rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
            soft_min = -gamma * (math.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + soft_min
    return float(R[len_a, len_b])


def spectral_angle(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity in the frequency domain expressed as a score in [0, 1]."""

    fft_hat = np.fft.rfft(y_hat, axis=-1)
    fft_true = np.fft.rfft(y, axis=-1)
    numerator = np.sum(np.conj(fft_hat) * fft_true).real
    denom = (np.linalg.norm(fft_hat) * np.linalg.norm(fft_true)) + _EPS
    cosine = numerator / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = math.acos(cosine)
    return 1.0 - angle / math.pi
