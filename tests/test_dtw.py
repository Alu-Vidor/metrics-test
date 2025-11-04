"""Tests for DTW-based structure metrics."""
from __future__ import annotations

import numpy as np

from src.metrics.structure import dtw_distance


def test_dtw_phase_invariance() -> None:
    base = np.sin(np.linspace(0, 2 * np.pi, 64))
    shifted = np.roll(base, 2)
    dtw_val = dtw_distance(base, shifted)
    l2_val = np.linalg.norm(base - shifted)
    assert dtw_val <= l2_val + 1e-6
