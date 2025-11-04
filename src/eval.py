"""Evaluation pipeline helpers."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


from .metrics.composite import composite, normalize


def evaluate(
    model: Any,
    test_data: Any,
    scenarios: Mapping[str, Any],
    metric_fns: Mapping[str, Callable[..., float]],
    normalizers: Mapping[str, str],
    component_weights: Mapping[str, float],
    lam: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a trained model on the provided scenarios."""

    if not hasattr(model, "predict"):
        raise AttributeError("Model must implement a predict method")

    y_true = test_data.targets
    predictions: Dict[str, Any] = {}
    transformed_inputs: Dict[str, Any] = {}
    for scenario_name, scenario in scenarios.items():
        transformed = scenario.apply(test_data.inputs)
        transformed_inputs[scenario_name] = transformed
        predictions[scenario_name] = model.predict(transformed)

    results: Dict[str, Dict[str, float]] = {}
    for scenario_name in scenarios:
        y_hat = predictions[scenario_name]

        acc_raw = metric_fns["accuracy"](y_hat, y_true)
        struct_raw = metric_fns["structure"](y_hat, y_true)
        rob_raw = metric_fns["robustness"](
            scenario_name=scenario_name,
            predictions_by_scenario=predictions,
            y_true=y_true,
        )

        acc = float(normalize([acc_raw], normalizers["accuracy"])[0])
        struct = float(normalize([struct_raw], normalizers["structure"])[0])
        rob = float(normalize([rob_raw], normalizers["robustness"])[0])

        cost_val = None
        if "cost" in metric_fns:
            cost_val = metric_fns["cost"](model=model, inputs=transformed_inputs[scenario_name])

        comp_score = composite(
            acc,
            struct,
            rob,
            alpha=component_weights.get("alpha", 1 / 3),
            beta=component_weights.get("beta", 1 / 3),
            gamma=component_weights.get("gamma", 1 / 3),
            cost=cost_val,
            lam=lam,
        )
        results[scenario_name] = {
            "accuracy": acc,
            "structure": struct,
            "robustness": rob,
            "cost": cost_val or 0.0,
            "composite": comp_score,
        }
    return results
