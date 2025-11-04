"""Utilities for loading time series data and preparing rolling windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TimeSeriesSplit:
    """Represents a contiguous slice of a time series dataset."""

    X: np.ndarray
    Y: np.ndarray
    indices: np.ndarray


@dataclass
class WindowedDataset:
    """Windowed representation suitable for supervised learning."""

    inputs: np.ndarray
    targets: np.ndarray
    indices: np.ndarray

    def __len__(self) -> int:
        return self.inputs.shape[0]


def make_splits(series: TimeSeriesSplit, t1: int, t2: int) -> Tuple[TimeSeriesSplit, TimeSeriesSplit, TimeSeriesSplit]:
    """Create chronological train/validation/test splits.

    Parameters
    ----------
    series:
        Full dataset represented as a :class:`TimeSeriesSplit` with arrays of
        shape ``(T, ...)``.
    t1, t2:
        Boundaries such that ``train=[0, t1)``, ``val=[t1, t2)`` and
        ``test=[t2, T)``.
    """

    if not 0 < t1 < t2 <= len(series.indices):
        raise ValueError("Expected 0 < t1 < t2 <= T")

    train_slice = slice(0, t1)
    val_slice = slice(t1, t2)
    test_slice = slice(t2, len(series.indices))

    return (
        TimeSeriesSplit(series.X[train_slice], series.Y[train_slice], series.indices[train_slice]),
        TimeSeriesSplit(series.X[val_slice], series.Y[val_slice], series.indices[val_slice]),
        TimeSeriesSplit(series.X[test_slice], series.Y[test_slice], series.indices[test_slice]),
    )


def make_windows(X: np.ndarray, Y: np.ndarray, L: int, H: int) -> WindowedDataset:
    """Construct sliding windows of length ``L`` and horizon ``H``.

    Returns
    -------
    WindowedDataset
        Dataset containing ``inputs`` of shape ``(N, L, F)`` where ``F`` is the
        number of features, ``targets`` of shape ``(N, H, ...)`` and
        ``indices`` tracking the original time positions.
    """

    if L <= 0 or H <= 0:
        raise ValueError("Window length and horizon must be positive")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("Feature and target lengths must match")

    T = X.shape[0]
    if T < L + H:
        raise ValueError("Time series too short for the requested window/horizon")

    num_windows = T - L - H + 1
    input_windows = []
    target_windows = []
    index_windows = []

    for end in range(L, L + num_windows):
        start = end - L
        horizon_end = end + H
        input_windows.append(X[start:end])
        target_windows.append(Y[end:horizon_end])
        index_windows.append((start, end, horizon_end))

    inputs = np.stack(input_windows, axis=0)
    targets = np.stack(target_windows, axis=0)
    indices = np.array(index_windows, dtype=np.int64)
    return WindowedDataset(inputs=inputs, targets=targets, indices=indices)
