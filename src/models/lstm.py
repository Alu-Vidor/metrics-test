"""Placeholder LSTM-style model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _default_mean(targets: np.ndarray) -> float:
    return float(np.mean(targets)) if targets.size else 0.0


@dataclass
class LSTMModel:
    hidden_size: int
    num_layers: int
    dropout: float = 0.0

    _mean: float = 0.0
    _horizon: int = 1

    def fit(self, dataset: Any) -> None:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError("Dataset must expose a 'targets' attribute")
        self._mean = _default_mean(np.asarray(targets))
        self._horizon = int(np.asarray(targets).shape[1])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch = inputs.shape[0]
        return np.full((batch, self._horizon), self._mean, dtype=float)
