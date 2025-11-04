"""Tests for the evaluation pipeline."""
from __future__ import annotations

import numpy as np

from src.eval import evaluate
from src.scenarios import Clean


class _DummyModel:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        batch, horizon, *_ = inputs.shape
        return np.full((batch, horizon), self.value, dtype=float)


class _DummyDataset:
    def __init__(self) -> None:
        self.inputs = np.zeros((4, 3, 1))
        self.targets = np.zeros((4, 3))
        self.indices = np.arange(4)


def test_evaluate_returns_cost_field() -> None:
    model = _DummyModel()
    dataset = _DummyDataset()
    scenarios = {"clean": Clean()}

    metric_fns = {
        "accuracy": lambda y_hat, y_true: 1.0,
        "structure": lambda y_hat, y_true: 1.0,
        "robustness": lambda **kwargs: 1.0,
        "cost": lambda **kwargs: 0.123,
    }

    normalizers = {"accuracy": "minmax", "structure": "minmax", "robustness": "minmax"}
    component_weights = {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3}

    results = evaluate(model, dataset, scenarios, metric_fns, normalizers, component_weights, lam=0.0)
    assert "clean" in results
    assert np.isclose(results["clean"]["cost"], 0.123)
