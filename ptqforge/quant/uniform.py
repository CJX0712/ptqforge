"""Uniform affine quantization (symmetric / asymmetric, per-tensor / per-channel).

Fake-quantization semantics: a float tensor is mapped to integers then back to
float, so downstream (numpy) forward passes consume the *dequantized* weights.
This is the standard way to evaluate post-training quantization accuracy.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import QuantConfig


def _int_range(cfg: QuantConfig):
    if cfg.symmetric:
        qmax = 2 ** (cfg.bits - 1) - 1
        return -qmax, qmax
    return 0, 2**cfg.bits - 1


def compute_qparams(w: np.ndarray, cfg: QuantConfig, axis=None):
    """Return (scale, zero_point) with broadcast-compatible shapes."""
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return np.array([1.0]), np.array([0.0])
    if axis is None:
        if cfg.symmetric:
            amax = float(np.max(np.abs(w)))
            qmax = 2 ** (cfg.bits - 1) - 1
            scale = amax / qmax if amax > 0 else 1.0
            return np.array([scale]), np.array([0.0])
        wmin, wmax = float(np.min(w)), float(np.max(w))
        qmax = 2**cfg.bits - 1
        scale = (wmax - wmin) / qmax if wmax > wmin else 1.0
        zp = float(np.clip(round(0.0 - wmin / scale), 0, qmax))
        return np.array([scale]), np.array([zp])
    # per-channel
    if cfg.symmetric:
        amax = np.max(np.abs(w), axis=axis, keepdims=True)
        qmax = 2 ** (cfg.bits - 1) - 1
        scale = np.where(amax > 0, amax / qmax, 1.0)
        return scale, np.zeros_like(scale)
    wmin = np.min(w, axis=axis, keepdims=True)
    wmax = np.max(w, axis=axis, keepdims=True)
    qmax = 2**cfg.bits - 1
    scale = np.where(wmax > wmin, (wmax - wmin) / qmax, 1.0)
    zp = np.round(0.0 - wmin / scale)
    zp = np.clip(zp, 0, qmax)
    return scale, zp


def fake_quantize(w: np.ndarray, cfg: QuantConfig, axis=None, clip_threshold=None):
    """Quantize-dequantize ``w``; return (dequantized_float, meta)."""
    w = np.asarray(w, dtype=np.float64)
    if clip_threshold is not None:
        w = np.clip(w, -float(clip_threshold), float(clip_threshold))
    scale, zp = compute_qparams(w, cfg, axis=axis)
    qmin, qmax = _int_range(cfg)
    q = np.round(w / scale + zp)
    # guard against overflow when scale is degenerate
    q = np.clip(q, qmin, qmax)
    wq = (q - zp) * scale
    meta = {"scale": scale, "zp": zp, "qmin": qmin, "qmax": qmax}
    return wq, meta


def quantize_tensor(w: np.ndarray, cfg: QuantConfig, clip_threshold=None):
    """Public entry: per-channel when ``cfg.per_channel`` else per-tensor."""
    axis = 0 if cfg.per_channel else None
    return fake_quantize(w, cfg, axis=axis, clip_threshold=clip_threshold)
