# PTQForge — Architecture

> Author: 晨星 · Domain: Post-Training Quantization (PTQ) · Grade: S

## 1. Goal & scope

Compress a **trained** model's weights to low precision without retraining,
recovering as much of the full-precision accuracy as possible at a reduced
average bit-width. PTQForge targets **weight-only** quantization on CPU, with a
pure-numpy, dependency-light implementation that is fully reproducible.

## 2. Method: QuantFuse (flagship)

A weight tensor `W` is quantized per output-channel with an **asymmetric affine**
scheme:

```
s_c = (max_c(W) - min_c(W)) / (2^b - 1)
zp_c = round(0 - min_c(W) / s_c)
q = clip(round(W_c / s_c + zp_c), 0, 2^b - 1)
Wq_c = (q - zp_c) * s_c
```

Four components compose QuantFuse:

1. **Per-channel asymmetric** — each output channel gets its own scale/zero-point,
   so channels with very different magnitudes are all represented well (vs a
   single shared per-tensor scale that starves small-magnitude channels).
2. **KL outlier clipping** — a TensorRT-style calibration searches a clipping
   threshold `T` that minimises the KL divergence between the full-distribution
   histogram and the clipped-then-quantized histogram, removing heavy-tail
   outliers that would otherwise blow up the shared scale.
3. **Mixed precision** — a HAWQ-style sensitivity pass quantizes each layer
   individually to INT4 and measures the score (accuracy / weight fidelity) drop;
   the `n_keep` most sensitive layers are kept at INT8 while the rest run INT4,
   giving an average bit-width of ~5 at near-INT8 accuracy.
4. **Adaptive rounding (AdaRound-style)** — soft-rounding minimises
   layer-output reconstruction error; guarded by an RTN fallback so it can never
   degrade below round-to-nearest.

## 3. Module layout (acyclic)

```
cli.py
  └─ pipeline.benchmark.QuantPipeline
        ├─ data.generators        (moons / circles / gaussian, seed-fixed)
        ├─ model.mlp              (numpy MLP, Adam — the quant target)
        ├─ quant.ptq              (orchestration + method registry)
        │     ├─ quant.uniform    (affine fake-quant)
        │     ├─ quant.calibration(KL clip threshold)
        │     ├─ quant.adaptive_rounding
        │     └─ quant.mixed_precision (sensitivity → bit alloc)
        ├─ eval.metrics           (SQNR / accuracy / significance)
        └─ backends               (optional torch/onnx probe, graceful skip)
              └─ core             (types, errors, config, seed)
```

`core` is the leaf: no internal import of the higher layers. This keeps every
module independently testable.

## 4. Determinism contract

- Single entry point `core.seed.set_all(seed)` sets `numpy` + `random` seeds.
- Datasets are generated in class-contiguous blocks and **shuffled before any
  train/test split** (a train/test leak was caught and fixed here: an index split
  on block-ordered data produced a non-representative, class-imbalanced holdout).
- `run_demo.py` runs the benchmark twice with the same master seed and asserts the
  core aggregate is **bit-for-bit identical** (elapsed time excluded).

## 5. No data leakage

- The calibration / sensitivity sets are drawn from the **training** partition
  only; the test partition is untouched until final scoring.
- Scaler / clipping thresholds are fit on train+calibration data, never on test.

## 6. Offline-fallback design

PTQForge is the Tier-1 numpy implementation itself. Optional accelerators
(`torch`, `onnxruntime`) are probed by `backends.available_*()`; when missing,
the pipeline proceeds on the numpy path and the report marks them `skipped`. No
metric is fabricated for an unavailable backend.

## 7. Performance budget

Demo end-to-end (3 datasets × 5 seeds + synthetic + ablation + failure analysis)
runs in **< 25 s** on CPU, peak memory well under 2 GB.
