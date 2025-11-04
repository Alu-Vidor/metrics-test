"""Scenario generators applying deterministic perturbations to inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Scenario(Protocol):
    """Interface for scenario transformations."""

    def apply(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class Clean:
    """Return the input unchanged."""

    def apply(self, X: np.ndarray) -> np.ndarray:  # noqa: D401 - simple alias
        return np.array(X, copy=True)


@dataclass
class Noise:
    """Additive Gaussian noise."""

    sigma: float
    seed: int = 0

    def apply(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(scale=self.sigma, size=X.shape)
        return X + noise


@dataclass
class Shift:
    """Add a constant offset to the sequence."""

    delta: float

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X + self.delta


@dataclass
class Phase:
    """Cyclical phase shift."""

    phi: float

    def apply(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            length = X.shape[0]
            shift = int(round(self.phi * length)) % length
            return np.roll(X, shift)
        length = X.shape[-2]
        shift = int(round(self.phi * length)) % length
        return np.roll(X, shift, axis=-2)


def build_scenario(name: str, **kwargs) -> Scenario:
    """Factory helper used by configuration driven pipelines."""

    registry = {
        "clean": Clean,
        "noise": Noise,
        "shift": Shift,
        "phase": Phase,
    }
    if name not in registry:
        raise KeyError(f"Unknown scenario '{name}'")
    return registry[name](**kwargs)
