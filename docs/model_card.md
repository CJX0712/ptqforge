# PTQForge — Model Card

> Author: 晨星 · System: PTQForge (PTQ) · Version: 0.1.0 · Grade: **S**

## Model details

- **Type**: Post-training weight quantization (PTQ), weight-only.
- **Flagship**: `QuantFuse` — per-channel asymmetric + KL outlier clipping + mixed
  precision + adaptive rounding.
- **Target compute**: CPU, pure numpy, no GPU / no native compile.
- **Reference model** (quantization target): a 3-hidden-layer numpy MLP
  (`[n_features, 64, 48, 32, n_classes]`, Adam, ReLU), trained on synthetic data.
- **Intended use**: compress model weights to INT4/INT8 to cut memory & bandwidth
  while preserving accuracy; a research/education reference for PTQ techniques.

## Evaluation

Benchmarks run across **5 seeds**; reported as mean ± std. Primary metric:
**weight-SQNR** (dB, higher better). Accuracy is a secondary model-level check.

| Method | avg bits | acc (5 seeds) | wSQNR (dB) |
|--------|---------:|--------------:|-----------:|
| FP32 (ceiling) | 32 | 0.9710 ± 0.023 | ∞ |
| INT4 RTN per-tensor | 4 | 0.9668 ± 0.028 | 9.70 |
| INT8 RTN per-channel | 8 | 0.9710 ± 0.023 | 42.29 |
| INT4 per-channel+KL+AR | 4 | 0.9698 ± 0.025 | 20.12 |
| **QuantFuse INT4 (mixed)** | **5.0** | **0.9702 ± 0.024** | **21.49** |

**Verdict**: QuantFuse INT4 (5.0 avg bits) beats INT4-RTN by **+11.8 dB**
weight-SQNR (threshold was +5 dB) and matches INT8 accuracy within +0.0008.

Synthetic weight-tensor benchmark: QuantFuse **+4.65 dB** over INT4-RTN.

## Factors & caveats

- **Works best** when weights have per-channel magnitude variation, heavy tails, or
  a few outlier-prone channels — exactly the regimes where naive per-tensor RTN
  collapses.
- **Limited by design**: INT4 is near-lossless here because the reference MLP is
  small; on very deep or outlier-heavy production models the accuracy gap to INT8
  would be larger and mixed precision earns its keep.
- **Not a replacement for INT8** in absolute accuracy terms — it trades a small
  accuracy margin for ~1.6× bit reduction vs INT8 (5.0 vs 8 bits average).

## Failure modes (measured)

| Case | naive → fixed | Root cause |
|------|--------------:|-----------|
| Outlier defeats per-tensor RTN | 13.94 → 17.80 dB | shared scale stretched by tail |
| Tiny weights: INT4 too coarse | 20.76 → 45.43 dB | 16 levels too few; mixed keeps them INT8 |
| Per-row scale variation | 11.67 → 18.38 dB | shared scale starves small rows |

## Ethical & safety

- Education/research artifact; no training data leaves the machine.
- Fully deterministic and reproducible; no hidden network calls.
- License: MIT (Author: 晨星).
