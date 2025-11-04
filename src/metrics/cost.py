"""Cost/efficiency related measurements."""
from __future__ import annotations

import time
import warnings
from typing import Any, Iterable

import numpy as np


def latency_ms(model: Any, inputs: np.ndarray, warmup: int = 3, iters: int = 30) -> float:
    """Measure prediction latency in milliseconds.

    The function performs ``warmup`` forward passes that are discarded followed
    by ``iters`` timed runs. The returned value corresponds to the average
    latency in milliseconds.
    """

    if not hasattr(model, "predict"):
        raise AttributeError("Model must implement a predict method")
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0")

    for _ in range(warmup):
        model.predict(inputs)

    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        model.predict(inputs)
        end = time.perf_counter()
        timings.append(end - start)

    mean_latency_ms = float(np.mean(timings) * 1000.0)
    return mean_latency_ms


def estimate_flops(model: Any, input_shape: Iterable[int]) -> float:
    """Best-effort FLOPs estimation.

    The current implementation serves as a placeholder and emits a warning,
    returning ``np.nan``. Users are encouraged to integrate framework-specific
    profilers (e.g. ``torch.profiler``) to obtain accurate measurements.
    """

    del model, input_shape
    warnings.warn(
        "FLOPs estimation is not implemented; returning NaN. Integrate a profiler "
        "for precise measurements.",
        RuntimeWarning,
        stacklevel=2,
    )
    return float("nan")


def normalize_cost(latency: float, reference_latency: float) -> float:
    """Normalize latency relative to a reference budget."""

    reference = reference_latency if reference_latency > 0 else 1.0
    return float(latency / reference)
