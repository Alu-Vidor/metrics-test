"""Robustness metrics composed from multiple stability components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping

import numpy as np


_EPS = 1e-8


def stability_score(
    y_hat_clean: np.ndarray,
    y_hat_perturbed: np.ndarray,
    y_true: np.ndarray,
    base_metric: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Return a normalized stability score in ``[0, 1]``.

    The score measures the relative degradation of ``base_metric`` when moving
    from the clean predictions to the perturbed ones. A value close to ``1``
    indicates that the perturbation does not significantly deteriorate the base
    metric, whereas values near ``0`` correspond to large drops in quality.
    """

    clean_score = float(base_metric(y_hat_clean, y_true))
    perturbed_score = float(base_metric(y_hat_perturbed, y_true))

    if not np.isfinite(clean_score):
        clean_score = 0.0
    if not np.isfinite(perturbed_score):
        perturbed_score = 0.0

    drop = max(0.0, clean_score - perturbed_score)
    denom = abs(clean_score) + _EPS
    stability = 1.0 - drop / denom
    return float(np.clip(stability, 0.0, 1.0))


@dataclass
class ScenarioPair:
    """Helper describing a perturbation scenario and its clean reference."""

    perturbed: str
    reference: str


def _normalize_weights(weights: Mapping[str, float]) -> MutableMapping[str, float]:
    total = float(sum(max(v, 0.0) for v in weights.values()))
    if total <= 0:
        return {name: 0.0 for name in weights}
    return {name: max(v, 0.0) / total for name, v in weights.items()}


def robustness(
    predictions_by_scenario: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    base_metric: Callable[[np.ndarray, np.ndarray], float],
    scenario_pairs: Mapping[str, tuple[str, str]] | Mapping[str, ScenarioPair] = {
        "noise": ("noise", "clean"),
        "shift": ("shift", "clean"),
        "phase": ("phase", "clean"),
    },
    weights: Mapping[str, float] = {"noise": 1 / 3, "shift": 1 / 3, "phase": 1 / 3},
    calibration_component: float | None = None,
) -> float:
    """Aggregate robustness score across multiple perturbation scenarios.

    Parameters
    ----------
    predictions_by_scenario:
        Mapping from scenario name to the model predictions obtained under that
        scenario.
    y_true:
        Ground-truth targets shared by all scenarios.
    base_metric:
        Metric function returning a score in ``[0, 1]`` for clean predictions.
    scenario_pairs:
        Mapping from component name to a pair ``(perturbed, reference)``
        identifying entries in ``predictions_by_scenario``. The perturbed
        element is compared against the reference (typically ``"clean"``).
    weights:
        Relative contribution of each component. The values do not need to sum
        to ``1`` as they are normalized internally.
    calibration_component:
        Optional calibration contribution already scaled to ``[0, 1]``.
    """

    norm_weights = _normalize_weights(weights)
    if not norm_weights:
        return 0.0

    total = 0.0
    for name, weight in norm_weights.items():
        if weight == 0.0:
            continue
        pair = scenario_pairs.get(name)
        if pair is None:
            raise KeyError(f"Missing scenario pair for component '{name}'")
        if isinstance(pair, ScenarioPair):
            perturbed_key = pair.perturbed
            reference_key = pair.reference
        else:
            perturbed_key, reference_key = pair
        if perturbed_key not in predictions_by_scenario:
            raise KeyError(f"Unknown perturbed scenario '{perturbed_key}'")
        if reference_key not in predictions_by_scenario:
            raise KeyError(f"Unknown reference scenario '{reference_key}'")
        y_hat_clean = predictions_by_scenario[reference_key]
        y_hat_perturbed = predictions_by_scenario[perturbed_key]
        stability = stability_score(y_hat_clean, y_hat_perturbed, y_true, base_metric)
        total += weight * stability

    if calibration_component is not None:
        calibration_component = float(np.clip(calibration_component, 0.0, 1.0))
        total = (total + calibration_component) / (sum(norm_weights.values()) + 1.0)
    return float(np.clip(total, 0.0, 1.0))
