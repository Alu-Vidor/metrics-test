"""End-to-end experiment runner driven by configuration files."""
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from .dataio import TimeSeriesSplit, make_splits, make_windows
from .eval import evaluate
from .metrics import accuracy as accuracy_metrics
from .metrics import composite as composite_metrics
from .metrics import structure as structure_metrics
from .metrics.cost import latency_ms, normalize_cost
from .metrics.robustness import robustness as robustness_score
from .report import summary_html, tables_to_csv
from .scenarios import build_scenario
from .train import train_model
from .weight_search import find_weights


LOGGER = logging.getLogger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset_passport(reference: str, *, base_path: Path | None = None) -> Dict[str, Any]:
    if "::" in reference:
        cfg_path, key = reference.split("::", 1)
    else:
        cfg_path, key = reference, "default"
    cfg_path_obj = Path(cfg_path)
    candidate_paths = []
    if base_path is not None and not cfg_path_obj.is_absolute():
        candidate_paths.append((base_path / cfg_path_obj).resolve())
    candidate_paths.append(cfg_path_obj.resolve() if cfg_path_obj.is_absolute() else cfg_path_obj)
    if not cfg_path_obj.is_absolute():
        candidate_paths.append((Path(__file__).resolve().parent / cfg_path_obj).resolve())
        candidate_paths.append((Path.cwd() / cfg_path_obj).resolve())

    seen = set()
    resolved_path = None
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            resolved_path = candidate
            break

    if resolved_path is None:
        searched = ", ".join(str(path) for path in seen)
        raise FileNotFoundError(f"Dataset passport file '{cfg_path}' not found. Searched: {searched}")

    data = _load_yaml(resolved_path)
    if key not in data:
        raise KeyError(f"Dataset passport '{key}' not found in {resolved_path}")
    return data[key]


def _minimum_series_length(window: int, horizon: int) -> int:
    """Compute minimal total length so each split can produce windows."""
    min_segment = window + horizon
    if min_segment <= 0:
        raise ValueError("Window and horizon must be positive")

    total = max(min_segment, 1)
    while True:
        t1 = int(0.6 * total)
        t2 = int(0.8 * total)
        train_len = t1
        val_len = t2 - t1
        test_len = total - t2
        if min(train_len, val_len, test_len) >= min_segment:
            return total
        total += 1


def _generate_synthetic_series(length: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    signal = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
    trend = 0.001 * t
    noise = rng.normal(scale=0.1, size=length)
    return signal + trend + noise


def _load_timeseries(dataset_cfg: Mapping[str, Any], fallback_length: int = 500) -> TimeSeriesSplit:
    path = Path(dataset_cfg["path"])
    if path.exists():
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        target_col = dataset_cfg.get("target") or dataset_cfg.get("column")
        if target_col is None:
            raise KeyError("Dataset configuration must specify a target column")
        values = frame[target_col].to_numpy(dtype=np.float32)
    else:
        LOGGER.warning("Dataset '%s' not found. Falling back to synthetic data.", path)
        values = _generate_synthetic_series(fallback_length)

    X = values[:, None]
    Y = values[:, None]
    indices = np.arange(len(values))
    return TimeSeriesSplit(X=X, Y=Y, indices=indices)


def _instantiate_model(module_path: str, model_name: str, params: Mapping[str, Any]) -> Any:
    module = importlib.import_module(module_path)
    parts = model_name.split("_")
    candidates = []
    camel = "".join(part.capitalize() for part in parts)
    if camel:
        candidates.append(f"{camel}Model")
    acronym = "".join(part.upper() for part in parts)
    if acronym and f"{acronym}Model" not in candidates:
        candidates.append(f"{acronym}Model")
    if camel and camel not in candidates:
        candidates.append(camel)
    if acronym and acronym not in candidates:
        candidates.append(acronym)
    candidates.append(model_name)

    cls = None
    for name in candidates:
        if hasattr(module, name):
            cls = getattr(module, name)
            break

    if cls is None:
        expected = ", ".join(candidates)
        raise AttributeError(f"Module '{module_path}' does not provide any of: {expected}")
    return cls(**params)


def _build_scenarios(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]]]:
    scenarios = {}
    scenario_pairs: Dict[str, Tuple[str, str]] = {}
    clean_name = None
    for spec in config:
        spec = dict(spec)
        scenario_type = spec.pop("type")
        name = spec.pop("name", None)
        if name is None:
            if scenario_type == "clean":
                name = "clean"
            elif len(spec) == 1:
                value = next(iter(spec.values()))
                name = f"{scenario_type}_{value}"
            else:
                params_str = "_".join(f"{k}={v}" for k, v in spec.items())
                name = f"{scenario_type}{('_' + params_str) if params_str else ''}"
        scenarios[name] = build_scenario(scenario_type, **spec)
        if scenario_type == "clean":
            clean_name = name

    if clean_name is None:
        raise ValueError("At least one clean scenario is required")

    for name, scenario in scenarios.items():
        if name == clean_name:
            continue
        scenario_pairs[name] = (name, clean_name)
    return scenarios, scenario_pairs


def _make_metric_functions(
    base_metric: Callable[[np.ndarray, np.ndarray], float],
    scenario_pairs: Mapping[str, Tuple[str, str]],
    scenario_weights: Mapping[str, float],
    cost_reference: float | None,
    calibration_component: float | None = None,
):
    clean_candidates = {pair[1] for pair in scenario_pairs.values()}
    clean_name = next(iter(clean_candidates), "clean")

    def robustness_metric(*, scenario_name: str, predictions_by_scenario: Mapping[str, np.ndarray], y_true: np.ndarray) -> float:
        if scenario_name == clean_name:
            return robustness_score(
                predictions_by_scenario=predictions_by_scenario,
                y_true=y_true,
                base_metric=base_metric,
                scenario_pairs=scenario_pairs,
                weights=scenario_weights,
                calibration_component=calibration_component,
            )
        if scenario_name not in scenario_pairs:
            return 1.0
        local_pairs = {scenario_name: scenario_pairs[scenario_name]}
        local_weights = {scenario_name: scenario_weights.get(scenario_name, 1.0)}
        return robustness_score(
            predictions_by_scenario=predictions_by_scenario,
            y_true=y_true,
            base_metric=base_metric,
            scenario_pairs=local_pairs,
            weights=local_weights,
            calibration_component=None,
        )

    metric_fns = {
        "accuracy": base_metric,
        "structure": structure_metrics.spectral_angle,
        "robustness": robustness_metric,
    }

    if cost_reference is not None:
        def cost_fn(*, model: Any, inputs: np.ndarray) -> float:
            raw_latency = latency_ms(model, inputs)
            return normalize_cost(raw_latency, cost_reference)

        metric_fns["cost"] = cost_fn

    return metric_fns


def _base_metric_factory(base_metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if base_metric == "smape":
        return accuracy_metrics.smape
    if base_metric.startswith("pinball_tau_"):
        tau = float(base_metric.split("pinball_tau_")[1]) / 100.0
        from .metrics.probabilistic import pinball

        return lambda y_hat, y_true: pinball(y_true, y_hat, tau=tau)
    return lambda y_hat, y_true: accuracy_metrics.lp_error(y_hat, y_true, p=1)


def run_experiment(config_path: Path) -> None:
    config = _load_yaml(config_path)
    seed = int(config.get("seed", 0))
    np.random.seed(seed)

    scenario_specs = config.get("scenarios", [])
    scenarios, scenario_pairs = _build_scenarios(scenario_specs)

    horizon = int(config.get("horizon", 24))
    window = int(config.get("window", 96))

    datasets_config = config.get("datasets", [])
    component_weights = config.get("component_weights", {"alpha": 1 / 3, "beta": 1 / 3, "gamma": 1 / 3})
    normalizers = config.get("normalization", {"accuracy": "minmax", "structure": "minmax", "robustness": "minmax"})
    scenario_weights_config = config.get("scenario_weights", {})
    lam = float(config.get("lambda", 0.0))
    cost_reference = config.get("budget", {}).get("target_latency_ms")

    base_metric_name = config.get("base_metric", "smape")
    base_metric = _base_metric_factory(base_metric_name)

    val_tables: Dict[str, Dict[str, Dict[str, float]]] = {}
    test_tables: Dict[str, Dict[str, Dict[str, float]]] = {}

    report_items = {}

    for dataset_entry in datasets_config:
        dataset_cfg = dict(dataset_entry)
        dataset_name = dataset_cfg.get("name", "dataset")
        passport_ref = dataset_cfg.get("config")
        if passport_ref:
            passport = _load_dataset_passport(passport_ref, base_path=config_path.parent)
            dataset_cfg.setdefault("target", passport.get("target"))
            horizon = passport.get("horizon", horizon)
            window = passport.get("window", window)

        min_total_length = _minimum_series_length(window, horizon)
        series = _load_timeseries(dataset_cfg, fallback_length=max(500, min_total_length))
        if len(series.indices) < min_total_length:
            raise ValueError(
                f"Dataset '{dataset_name}' has length {len(series.indices)}, "
                f"but at least {min_total_length} points are required for window={window}, horizon={horizon}."
            )
        total = len(series.indices)
        t1 = int(0.6 * total)
        t2 = int(0.8 * total)
        train_split, val_split, test_split = make_splits(series, t1, t2)

        train_windows = make_windows(train_split.X, train_split.Y, window, horizon)
        val_windows = make_windows(val_split.X, val_split.Y, window, horizon)
        test_windows = make_windows(test_split.X, test_split.Y, window, horizon)

        robustness_weights = {name: scenario_weights_config.get(name, 1.0) for name in scenario_pairs}

        metric_fns = _make_metric_functions(
            base_metric=base_metric,
            scenario_pairs=scenario_pairs,
            scenario_weights=robustness_weights,
            cost_reference=cost_reference,
        )

        model_results = {}
        for model_spec in config.get("models", []):
            model_name = model_spec["name"]
            LOGGER.info("Training model %s on dataset %s", model_name, dataset_name)
            model = _instantiate_model(model_spec["module"], model_name, model_spec.get("params", {}))
            training = train_model(model, train_windows, val_windows, base_metric_name)
            LOGGER.info("Best checkpoint saved to %s", training.best_checkpoint)

            val_result = evaluate(
                model,
                val_windows,
                scenarios,
                metric_fns,
                normalizers,
                component_weights,
                lam,
            )
            test_result = evaluate(
                model,
                test_windows,
                scenarios,
                metric_fns,
                normalizers,
                component_weights,
                lam,
            )

            val_tables[model_name] = val_result
            test_tables[model_name] = test_result

            tables_to_csv(test_result, Path("outputs/reports") / f"{dataset_name}_{model_name}.csv")

            model_results[model_name] = {
                "validation": pd.DataFrame.from_dict(val_result, orient="index"),
                "test": pd.DataFrame.from_dict(test_result, orient="index"),
                "training": pd.DataFrame([training.metrics]),
            }

        for model_name, frames in model_results.items():
            report_items[f"{dataset_name}::{model_name}::validation"] = frames["validation"]
            report_items[f"{dataset_name}::{model_name}::test"] = frames["test"]
            report_items[f"{dataset_name}::{model_name}::training"] = frames["training"]

    best_alpha, best_beta, best_gamma = find_weights(val_tables, config.get("scenario_weights", {}), lam, grid_step=config.get("weight_search", {}).get("grid_step", 0.05))
    LOGGER.info("Selected weights: alpha=%.3f, beta=%.3f, gamma=%.3f", best_alpha, best_beta, best_gamma)

    summary_rows = []
    scenario_weights = config.get("scenario_weights", {})
    for model_name, scenario_table in test_tables.items():
        updated = {}
        for scenario_name, metrics in scenario_table.items():
            comp = composite_metrics.composite(
                metrics["accuracy"],
                metrics["structure"],
                metrics["robustness"],
                alpha=best_alpha,
                beta=best_beta,
                gamma=best_gamma,
                cost=metrics.get("cost"),
                lam=lam,
            )
            updated[scenario_name] = dict(metrics)
            updated[scenario_name]["composite"] = comp
        test_tables[model_name] = updated

        weighted_score = 0.0
        weight_sum = 0.0
        for scenario_name, metrics in updated.items():
            weight = scenario_weights.get(scenario_name, 1.0)
            weighted_score += weight * metrics["composite"]
            weight_sum += weight
        weighted_score /= max(weight_sum, 1.0)
        summary_rows.append({"model": model_name, "weighted_composite": weighted_score})

    summary_df = pd.DataFrame(summary_rows).sort_values("weighted_composite", ascending=False)
    report_items["Selected weights"] = pd.DataFrame({"alpha": [best_alpha], "beta": [best_beta], "gamma": [best_gamma]})
    report_items["Model ranking"] = summary_df

    summary_html(report_items, Path("outputs/reports/summary.html"))

    LOGGER.info("Experiment completed. Summary written to outputs/reports/summary.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full evaluation pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
