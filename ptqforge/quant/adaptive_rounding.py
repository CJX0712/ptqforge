"""Adaptive rounding (AdaRound-style) for weight quantization.

Minimizes per-layer reconstruction error ``||W a - Wq a||^2`` over a small
calibration batch ``a`` (layer inputs). A soft-rounding variable ``h`` is
annealed from continuous (RTN) toward {0,1} with a lasso regularizer pushing
weights toward integer rounding.

Safety: the result is compared against plain RTN on SQNR and the better one is
kept, so adaptive rounding can never degrade a layer below RTN.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from ..core.types import QuantConfig
from .uniform import compute_qparams, fake_quantize


def adaptive_round(
    w: np.ndarray,
    A: np.ndarray,
    cfg: QuantConfig,
    clip_threshold=None,
    iters=None,
    reg=None,
    gamma_init=None,
    gamma_end=None,
) -> np.ndarray:
    """Return dequantized weights after adaptive rounding (RTN-fallback guarded)."""
    w = np.asarray(w, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 2 and A.shape[0] != w.shape[1]:
        A = A.T  # ensure (in_features, batch)
    if A.size == 0:
        return fake_quantize(w, cfg, axis=0, clip_threshold=clip_threshold)[0]

    iters = cfg.ar_iters if iters is None else iters
    reg = cfg.ar_reg if reg is None else reg
    gamma_init = cfg.ar_gamma_init if gamma_init is None else gamma_init
    gamma_end = cfg.ar_gamma_end if gamma_end is None else gamma_end
    if iters <= 0:
        return fake_quantize(w, cfg, axis=0, clip_threshold=clip_threshold)[0]

    wc = w if clip_threshold is None else np.clip(w, -clip_threshold, clip_threshold)
    scale, zp = compute_qparams(wc, cfg, axis=0)
    w_floor = np.floor(wc / scale + zp)
    diff = (wc / scale) - w_floor  # in [0,1]
    h = diff.copy()

    lr = 0.1
    for t in range(iters):
        gamma = gamma_init + (gamma_end - gamma_init) * (t / max(1, iters - 1))
        sig = 1.0 / (1.0 + np.exp(-gamma * (h - 0.5)))
        wq = (zp + w_floor + sig) * scale
        err = (wc - wq) @ A  # (out, batch)
        dLdWq = -(err @ A.T)  # (out, in)
        sig_prime = gamma * sig * (1.0 - sig)
        grad = dLdWq * scale * sig_prime + reg * np.sign(h - 0.5)
        h = np.clip(h - lr * grad, 0.0, 1.0)

    sig_final = 1.0 / (1.0 + np.exp(-gamma_end * (h - 0.5)))
    final_h = (sig_final > 0.5).astype(np.float64)
    wq_final = (zp + w_floor + final_h) * scale

    rtn = fake_quantize(w, cfg, axis=0, clip_threshold=clip_threshold)[0]
    rtn_sqnr = _sqnr(w, rtn)
    ar_sqnr = _sqnr(w, wq_final)
    return wq_final if ar_sqnr >= rtn_sqnr else rtn


def _sqnr(x: np.ndarray, xq: np.ndarray) -> float:
    x = x.ravel()
    xq = xq.ravel()
    denom = float(np.sum((x - xq) ** 2))
    if denom <= 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(x**2) / denom))
