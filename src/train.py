"""Training utilities for time series forecasting models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingResult:
    """Stores metadata from a training run."""

    best_checkpoint: str
    metrics: Dict[str, float]


def train_model(model: Any, train_data: Any, val_data: Any, base_loss: str) -> TrainingResult:
    """Placeholder training loop.

    Real implementations should hook into the user's ML framework of choice.
    This function documents the expected contract for downstream evaluation.
    """

    # In a real implementation, optimise the model here and persist checkpoints.
    del model, train_data, val_data, base_loss
    return TrainingResult(best_checkpoint="path/to/checkpoint.pt", metrics={})
