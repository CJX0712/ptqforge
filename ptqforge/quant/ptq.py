"""PTQ orchestration: per-layer quantization and the QuantFuse flagship.

``QuantFuse`` (混合精度旗舰) combines:
  * per-channel **asymmetric** affine quantization,
  * **KL-divergence outlier clipping** (TensorRT-style calibration),
  * **mixed precision** (sensitive layers kept INT8, rest INT4),
  * optional **adaptive rounding** (AdaRound-style, RTN-fallback guarded).

Reference / SOTA baseline is Round-To-Nearest (Jacob et al., 2017) with
per-channel symmetric quantization + activation calibration (TensorRT / PyTorch
Quantization). We reimplement these faithfully; we do not claim to beat
full-precision, only to beat the RTN baselines.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from ..core.types import QuantConfig
from .adaptive_rounding import adaptive_round
from .calibration import clip_threshold_for
from .mixed_precision import assign_mixed_bits
from .uniform import fake_quantize


def quantize_layer(w: np.ndarray, cfg: QuantConfig, a_in: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantize-dequantize one weight tensor under ``cfg``."""
    thr = clip_threshold_for(w, cfg)
    if cfg.adaptive_rounding and a_in is not None:
        return adaptive_round(w, a_in, cfg, clip_threshold=thr)
    axis = 0 if cfg.per_channel else None
    return fake_quantize(w, cfg, axis=axis, clip_threshold=thr)[0]


def method_configs() -> Dict[str, dict]:
    """Registry of evaluated methods.

    Each entry: cfg (QuantConfig) + flags (fp32 / mixed / n_keep).
    """
    return {
        "fp32": {
            "cfg": QuantConfig(
                bits=32, symmetric=True, per_channel=False, clip_outliers=False, adaptive_rounding=False
            ),
            "fp32": True,
        },
        "int4_rtn_pt": {
            "cfg": QuantConfig(bits=4, symmetric=True, per_channel=False, clip_outliers=False, adaptive_rounding=False),
        },
        "int8_rtn_pc": {
            "cfg": QuantConfig(bits=8, symmetric=True, per_channel=True, clip_outliers=False, adaptive_rounding=False),
        },
        "int4_pc": {
            "cfg": QuantConfig(bits=4, symmetric=False, per_channel=True, clip_outliers=True, adaptive_rounding=True),
        },
        "quantfuse_int4": {
            "cfg": QuantConfig(bits=4, symmetric=False, per_channel=True, clip_outliers=True, adaptive_rounding=True),
            "mixed": True,
            "n_keep": 1,
        },
    }


def quantfuse_bits(
    weights: List[np.ndarray],
    per_layer_input: List[np.ndarray],
    score_fn: callable,
    cfg: QuantConfig,
    n_keep: int = 1,
) -> tuple[List[int], List[float]]:
    """Assign mixed bits via sensitivity (flagship helper)."""

    def _q(w, bits, a_in):
        return quantize_layer(w, replace(cfg, bits=bits), a_in=a_in)

    return assign_mixed_bits(weights, per_layer_input, _q, score_fn, n_keep=n_keep)


def quantize_model_mixed(
    weights: List[np.ndarray],
    per_layer_input: List[np.ndarray],
    cfg: QuantConfig,
    bits: List[int],
) -> List[np.ndarray]:
    """Quantize each layer with its assigned bit-width (same scheme)."""
    out = []
    for i, w in enumerate(weights):
        out.append(quantize_layer(w, replace(cfg, bits=bits[i]), a_in=per_layer_input[i]))
    return out


def quantize_model_uniform(
    weights: List[np.ndarray],
    cfg: QuantConfig,
) -> List[np.ndarray]:
    """Quantize every layer identically (non-mixed methods)."""
    return [quantize_layer(w, cfg) for w in weights]
