"""Deterministic synthetic data generators (pure numpy, offline).

Two families are provided so the benchmark exercises both a linearly
separable regime (Gaussian blobs) and a non-linear regime (two moons):

* ``make_gaussian_blobs`` -- controlled class separation ``sep`` (difficulty knob).
* ``make_moons``         -- inherently non-linear, needs a real MLP.

All generators accept an explicit ``rng`` (``np.random.RandomState``) so the
whole benchmark is reproducible. No external datasets are downloaded.

Author: 晨星
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from ..core.errors import DataError

Dataset = Tuple[np.ndarray, np.ndarray]


def make_gaussian_blobs(
    rng: np.random.RandomState,
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 2,
    sep: float = 2.0,
    class_std: float = 1.0,
) -> Dataset:
    """Gaussian blobs on a hyper-sphere of radius ``sep``.

    Larger ``sep`` -> easier; smaller ``sep`` -> harder (bigger quantization gap).
    """
    if n_classes < 2:
        raise DataError("n_classes must be >= 2")
    # place class centers evenly on a circle in the first 2 dims, rest 0
    angles = np.linspace(0.0, 2.0 * np.pi, n_classes, endpoint=False)
    centers = np.zeros((n_classes, n_features), dtype=np.float64)
    centers[:, 0] = np.cos(angles) * sep
    centers[:, 1] = np.sin(angles) * sep
    y = rng.randint(0, n_classes, size=n_samples)
    X = centers[y] + rng.randn(n_samples, n_features) * class_std
    return X.astype(np.float64), y.astype(np.int64)


def make_moons(
    rng: np.random.RandomState,
    n_samples: int = 2000,
    noise: float = 0.15,
) -> Dataset:
    """Canonical interlocking two-moons (non-linear, needs a real MLP)."""
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    x0 = np.c_[np.cos(t), np.sin(t)]
    x1 = np.c_[1.0 - np.cos(t), 1.0 - np.sin(t)]
    X = np.r_[x0, x1] + rng.randn(2 * n, 2) * noise
    y = np.r_[np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]
    return X.astype(np.float64), y


def make_circles(
    rng: np.random.RandomState,
    n_samples: int = 2000,
    noise: float = 0.10,
    factor: float = 0.5,
) -> Dataset:
    """Concentric circles (non-linear, XOR-like; needs a hidden layer)."""
    n = n_samples // 2
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r_outer = 1.0 + rng.randn(n) * noise
    r_inner = factor + rng.randn(n) * noise
    x_outer = np.c_[np.cos(theta) * r_outer, np.sin(theta) * r_outer]
    x_inner = np.c_[np.cos(theta) * r_inner, np.sin(theta) * r_inner]
    X = np.r_[x_outer, x_inner]
    y = np.r_[np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]
    return X.astype(np.float64), y


# Registry consumed by the pipeline. Each builder returns (X, y) for a given rng.
DATASETS: Dict[str, Callable[[np.random.RandomState], Dataset]] = {
    "gaussian": lambda rng: make_gaussian_blobs(rng, sep=2.0),
    "moons": lambda rng: make_moons(rng, noise=0.10),
    "circles": lambda rng: make_circles(rng, noise=0.08),
}


def get_dataset(name: str, rng: np.random.RandomState) -> Dataset:
    if name not in DATASETS:
        raise DataError(f"unknown dataset {name!r}; available={sorted(DATASETS)}")
    return DATASETS[name](rng)
