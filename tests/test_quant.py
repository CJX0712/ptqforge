"""Tests for quantization primitives.

Author: 晨星
"""

import numpy as np

from ptqforge.core.types import QuantConfig
from ptqforge.eval.metrics import sqnr
from ptqforge.quant.adaptive_rounding import adaptive_round
from ptqforge.quant.calibration import kl_threshold
from ptqforge.quant.mixed_precision import assign_mixed_bits
from ptqforge.quant.uniform import fake_quantize


def _gaussian(shape=(32, 32), seed=0):
    return np.random.RandomState(seed).standard_normal(shape)


def test_int8_symmetric_per_tensor_is_lossless():
    w = _gaussian()
    cfg = QuantConfig(bits=8, symmetric=True, per_channel=False)
    wq, _ = fake_quantize(w, cfg)
    assert sqnr(w, wq) > 35.0


def test_per_channel_beats_per_tensor_on_scale_variation():
    rng = np.random.RandomState(1)
    # one channel lives on a completely different magnitude (typical in real
    # weight tensors) -> per-tensor shares a single scale and wastes the rest.
    base = rng.standard_normal((16, 16))
    w = base.copy()
    w[0] = base[0] * 200.0
    cfg_pt = QuantConfig(bits=4, symmetric=True, per_channel=False)
    cfg_pc = QuantConfig(bits=4, symmetric=True, per_channel=True)

    def mean_row_sqnr(ref, q):
        rows = []
        for r in range(ref.shape[0]):
            s = float(np.sum(ref[r] ** 2))
            n = float(np.sum((ref[r] - q[r]) ** 2))
            rows.append(10.0 * np.log10(s / n) if n > 0 else float("inf"))
        return float(np.mean(rows))

    sq_pt = mean_row_sqnr(w, fake_quantize(w, cfg_pt, axis=None)[0])
    sq_pc = mean_row_sqnr(w, fake_quantize(w, cfg_pc, axis=0)[0])
    # per-channel gives every channel its own (optimal) scale -> every channel's
    # reconstruction is at least as good, so the mean row SQNR must improve.
    assert sq_pc > sq_pt + 5.0


def test_kl_threshold_finite_and_positive():
    rng = np.random.RandomState(2)
    w = np.concatenate([rng.standard_normal(1000), rng.standard_normal(50) * 20.0])  # outliers
    thr = kl_threshold(np.abs(w), num_bins=512)
    assert np.isfinite(thr) and thr > 0


def test_adaptive_rounding_never_degrades():
    rng = np.random.RandomState(3)
    w = rng.standard_normal((16, 12))
    A = rng.standard_normal((12, 32))  # layer inputs (in_features, batch)
    cfg = QuantConfig(bits=4, symmetric=False, per_channel=True, adaptive_rounding=True)
    rtn = fake_quantize(w, cfg, axis=0)[0]
    ar = adaptive_round(w, A, cfg)
    assert sqnr(w, ar) >= sqnr(w, rtn) - 1e-9


def test_mixed_precision_keeps_n_keep_high_bits():
    rng = np.random.RandomState(4)
    # layer 2 has 50x larger weights -> quantizing it to INT4 hurts far more
    weights = [rng.standard_normal((8, 8)) * (50.0 if i == 2 else 1.0) for i in range(4)]
    orig = weights

    def quantize_fn(w, bits, a_in):
        cfg = QuantConfig(bits=bits, symmetric=True, per_channel=True)
        return fake_quantize(w, cfg, axis=0)[0]

    def score_fn(Ws):
        # closeness to the original full-precision weights (higher is better)
        err = 0.0
        for i in range(len(Ws)):
            d = Ws[i] - orig[i]
            err += float(np.sum(d * d))
        return -err

    bits, sens = assign_mixed_bits(weights, [None] * 4, quantize_fn, score_fn, n_keep=1)
    assert bits.count(8) == 1
    assert bits[2] == 8
    assert len(bits) == 4
