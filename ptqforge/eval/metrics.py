"""Evaluation metrics and aggregation helpers.

Author: 晨星
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def sqnr(x: np.ndarray, xq: np.ndarray) -> float:
    """Signal-to-Quantization-Noise Ratio in dB (higher = better)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    xq = np.asarray(xq, dtype=np.float64).ravel()
    denom = float(np.sum((x - xq) ** 2))
    if denom <= 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(x**2) / denom))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def weight_sqnr(weights: Sequence[np.ndarray], wq: Sequence[np.ndarray]) -> float:
    """Aggregate element-weighted weight SQNR across layers (dB)."""
    num = 0.0
    den = 0.0
    for a, b in zip(weights, wq, strict=True):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        num += float(np.sum(a**2))
        den += float(np.sum((a - b) ** 2))
    if den <= 0:
        return float("inf")
    return float(10.0 * np.log10(num / den))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """True iff mean_a - mean_b exceeds half the summed std (skill threshold)."""
    return (mean_a - mean_b) > 0.5 * (std_a + std_b)
