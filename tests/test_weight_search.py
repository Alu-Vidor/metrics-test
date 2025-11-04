"""Tests for weight search logic."""
from __future__ import annotations

from src.weight_search import find_weights


def test_find_weights_prefers_accuracy_when_dominant() -> None:
    val_tables = {
        "model_a": {
            "clean": {"accuracy": 0.9, "structure": 0.5, "robustness": 0.4, "cost": 0.0},
            "noise": {"accuracy": 0.85, "structure": 0.45, "robustness": 0.35, "cost": 0.0},
        }
    }
    scenario_weights = {"clean": 1.0, "noise": 1.0}
    alpha, beta, gamma = find_weights(val_tables, scenario_weights, lam=0.0, grid_step=0.5)
    assert alpha > beta
    assert alpha > gamma
