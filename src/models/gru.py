"""Placeholder GRU-style model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _default_last(targets: np.ndarray) -> float:
    return float(targets[..., -1].mean()) if targets.size else 0.0


@dataclass
class GRUModel:
    hidden_size: int
    num_layers: int
    dropout: float = 0.0

    _last: float = 0.0
    _horizon: int = 1

    def fit(self, dataset: Any) -> None:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError("Dataset must expose a 'targets' attribute")
        arr = np.asarray(targets)
        self._last = _default_last(arr)
        self._horizon = int(arr.shape[1])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch = inputs.shape[0]
        return np.full((batch, self._horizon), self._last, dtype=float)
