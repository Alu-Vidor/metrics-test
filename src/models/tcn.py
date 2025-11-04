"""Placeholder Temporal Convolutional Network model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _default_linear(targets: np.ndarray) -> float:
    if targets.size == 0:
        return 0.0
    trend = np.linspace(0.0, 1.0, targets.shape[1])
    return float((targets @ trend).mean() / (trend.sum() + 1e-8))


@dataclass
class TCNModel:
    channels: int
    levels: int
    kernel_size: int
    dropout: float = 0.0

    _linear: float = 0.0
    _horizon: int = 1

    def fit(self, dataset: Any) -> None:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError("Dataset must expose a 'targets' attribute")
        arr = np.asarray(targets)
        self._linear = _default_linear(arr)
        self._horizon = int(arr.shape[1])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch = inputs.shape[0]
        horizon = self._horizon
        base = np.linspace(0.0, 1.0, horizon)
        return np.tile(base * self._linear, (batch, 1))
