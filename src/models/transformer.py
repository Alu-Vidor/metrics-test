"""Placeholder Transformer-style model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _default_median(targets: np.ndarray) -> float:
    return float(np.median(targets)) if targets.size else 0.0


@dataclass
class TransformerModel:
    d_model: int
    nhead: int
    num_layers: int
    dropout: float = 0.0

    _median: float = 0.0
    _horizon: int = 1

    def fit(self, dataset: Any) -> None:
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise AttributeError("Dataset must expose a 'targets' attribute")
        arr = np.asarray(targets)
        self._median = _default_median(arr)
        self._horizon = int(arr.shape[1])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch = inputs.shape[0]
        return np.full((batch, self._horizon), self._median, dtype=float)
