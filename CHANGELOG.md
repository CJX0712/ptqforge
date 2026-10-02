# Changelog

All notable changes to PTQForge are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to
semantic versioning.

## [0.1.0] — 2026-10-03

### Added
- **QuantFuse** flagship PTQ method: per-channel asymmetric affine quantization +
  KL-divergence outlier clipping (TensorRT-style calibration) + mixed precision
  (sensitive layers kept INT8 via HAWQ-style sensitivity) + adaptive rounding
  (AdaRound-style, with an RTN-fallback guard so it never degrades).
- Reference baselines: FP32 ceiling, INT4 RTN per-tensor (Jacob et al. 2017),
  INT8 RTN per-channel (TensorRT / PyTorch Quantization paradigm).
- `QuantPipeline`: end-to-end `benchmark()` over (dataset × seed), plus
  `benchmark_synthetic()`, `ablation_study()`, `analyze_failure_cases()`.
- Optional-backend probe (`backends/`) for `torch` / `onnxruntime` with graceful
  skip when absent (offline-fallback guarantee).
- Deterministic seeding via `core.seed.set_all(seed)`; bit-for-bit reproducibility
  verified by a second-run check in `run_demo.py`.
- 28 pytest tests (CLI smoke, determinism, offline fallback, per-component
  invariants); `ruff` clean; CI matrix 3.12/3.13.

### Performance (5 seeds)
- QuantFuse INT4 (5.0 avg bits) weight-SQNR **+11.8 dB** over INT4-RTN on the
  trained-model benchmark (21.49 vs 9.70 dB), and matches INT8 accuracy
  (0.9702 vs 0.9710).
- Synthetic weight-tensor benchmark: **+4.65 dB** over INT4-RTN.

### Author
- 晨星
