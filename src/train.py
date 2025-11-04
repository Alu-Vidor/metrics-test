"""Training utilities for time series forecasting models."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import torch


@dataclass
class TrainingResult:
    """Stores metadata from a training run."""

    best_checkpoint: str
    metrics: Dict[str, float]


def _resolve_loss(base_loss: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Map configuration strings to differentiable loss functions."""

    loss = base_loss.lower()
    if loss == "mse":
        return lambda pred, target: torch.mean((pred - target) ** 2)
    if loss == "mae":
        return lambda pred, target: torch.mean(torch.abs(pred - target))
    if loss.startswith("pinball_tau_"):
        try:
            tau = float(loss.split("pinball_tau_")[1]) / 100.0
        except (IndexError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid pinball loss specification: {base_loss}") from exc

        def _pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            diff = target - pred
            return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))

        return _pinball
    raise ValueError(f"Unsupported base loss '{base_loss}'")


def train_model(model: Any, train_data: Any, val_data: Any, base_loss: str) -> TrainingResult:
    """Train ``model`` using the provided datasets and persist its weights."""

    if not hasattr(model, "fit"):
        raise AttributeError("Model must implement a fit method")

    loss_fn = _resolve_loss(base_loss)
    metrics = model.fit(train_data, val_data=val_data, loss_fn=loss_fn)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{uuid.uuid4().hex}.pt"

    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), checkpoint_path)
    else:  # pragma: no cover - fallback for custom models
        torch.save(getattr(model, "__dict__", {}), checkpoint_path)

    return TrainingResult(best_checkpoint=str(checkpoint_path), metrics=metrics)
