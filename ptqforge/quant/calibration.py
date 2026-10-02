"""Activation/weight range calibration.

Implements the TensorRT-style KL-divergence threshold search that clips
extreme outliers before quantization -- the canonical PTQ calibration step.
We search, over candidate clipping thresholds, the one whose quantized
distribution best matches the reference (full) distribution in KL divergence.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import QuantConfig


def _smooth_distribution(p: np.ndarray) -> np.ndarray:
    """Spread zero-mass across non-zero bins (avoids div-by-zero in KL)."""
    p = np.asarray(p, dtype=np.float64)
    is_zeros = (p == 0).astype(np.float64)
    n_zeros = int(is_zeros.sum())
    if n_zeros == 0:
        return p
    eps = np.finfo(np.float64).eps
    is_nonzeros = (p != 0).astype(np.float64)
    n_nonzeros = is_nonzeros.sum()
    mean = n_nonzeros / (p.size - n_zeros)
    zeros_to_distribute = mean * n_zeros
    new_p = p + is_nonzeros * zeros_to_distribute / (n_nonzeros + eps)
    return np.clip(new_p, eps, 1.0)


def _quantize_distribution(distribution: np.ndarray, num_quant_bins: int) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    num_bins = distribution.size
    if num_bins <= num_quant_bins:
        return distribution
    scale = num_bins / num_quant_bins
    new_distribution = np.zeros(num_quant_bins, dtype=np.float64)
    for i in range(num_quant_bins):
        start = int(np.floor(i * scale))
        end = int(np.ceil((i + 1) * scale))
        if end > start:
            new_distribution[i] = distribution[start:end].sum()
    return new_distribution


def _expand_quantized_distribution(distribution: np.ndarray, num_bins: int) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=np.float64)
    num_quant_bins = distribution.size
    scale = num_bins / num_quant_bins
    new_distribution = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_quant_bins):
        start = int(np.floor(i * scale))
        end = int(np.ceil((i + 1) * scale))
        if end > start:
            new_distribution[start:end] = distribution[i] / (end - start)
    return new_distribution


def _calculate_kl(p: np.ndarray, q: np.ndarray) -> float:
    if p.size > q.size:
        p = _quantize_distribution(p, q.size)
    elif q.size > p.size:
        q = _quantize_distribution(q, p.size)
    eps = np.finfo(np.float64).eps
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def kl_threshold(abs_w: np.ndarray, num_bins: int = 2048, num_quant_bins: int = 128, start_bin: int = 8) -> float:
    """Find the clipping threshold minimizing KL(P || Q) over candidate bins."""
    abs_w = np.abs(np.asarray(abs_w, dtype=np.float64)).ravel()
    if abs_w.size == 0:
        return 0.0
    absmax = float(abs_w.max())
    if absmax == 0:
        return 0.0
    hist, bin_edges = np.histogram(abs_w, bins=num_bins, range=(0.0, absmax))
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return absmax
    start_bin = max(start_bin, 1)
    n_candidates = num_bins - start_bin
    divergence = np.zeros(n_candidates)
    thresholds = np.zeros(n_candidates)
    for i, t in enumerate(range(start_bin, num_bins)):
        p = hist[: t + 1].copy()
        p[-1] += hist[t + 1 :].sum()
        q = _quantize_distribution(p, num_quant_bins)
        q = _expand_quantized_distribution(q, t + 1)
        p_n = _smooth_distribution(p)
        q_n = _smooth_distribution(q)
        divergence[i] = _calculate_kl(p_n, q_n)
        thresholds[i] = bin_edges[t]
    best = int(np.argmin(divergence))
    return float(thresholds[best])


def clip_threshold_for(w: np.ndarray, cfg: QuantConfig):
    """Return a clipping threshold (or None for minmax / disabled)."""
    if cfg.clip_outliers and cfg.calib_method == "kl":
        return kl_threshold(np.abs(w), num_bins=cfg.kl_bins)
    return None
