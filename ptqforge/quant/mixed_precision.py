"""Mixed-precision bit allocation by per-layer sensitivity.

HAWQ-style: quantize each layer individually to the low bit-width and measure
the resulting accuracy drop against the full-precision model. The most
sensitive ``n_keep`` layers are kept at higher precision (INT8); the rest run
at the target low precision (INT4). This recovers most of the INT8 accuracy at
a mostly-INT4 memory/compute budget.

Author: 晨星
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np


def assign_mixed_bits(
    weights: List[np.ndarray],
    per_layer_input: List[np.ndarray],
    quantize_fn: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
    score_fn: Callable[[List[np.ndarray]], float],
    n_keep: int = 1,
    low_bits: int = 4,
    high_bits: int = 8,
) -> tuple[List[int], List[float]]:
    """Return (bits_per_layer, sensitivity_per_layer).

    ``quantize_fn(w, bits, a_in)`` quantizes one layer; ``score_fn(Ws)`` scores
    a full weight list. Sensitivity is the absolute accuracy drop when only
    layer ``i`` is quantized to ``low_bits`` while everything else stays FP32.
    """
    acc_full = score_fn(weights)
    n = len(weights)
    sensitivity: List[float] = []
    for i in range(n):
        wq = list(weights)
        wq[i] = quantize_fn(weights[i], low_bits, per_layer_input[i])
        acc_i = score_fn(wq)
        sensitivity.append(abs(acc_full - acc_i))
    order = sorted(range(n), key=lambda i: -sensitivity[i])
    bits = [low_bits] * n
    for i in order[: max(0, min(n_keep, n))]:
        bits[i] = high_bits
    return bits, sensitivity
