"""PTQForge command-line entry point.

Author: 晨星
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ptqforge.backends import benchmark_with_optional_backends
from ptqforge.core.seed import set_all
from ptqforge.pipeline.benchmark import QuantPipeline

_METHOD_LABELS = {
    "fp32": "FP32 (ceiling)",
    "int4_rtn_pt": "INT4 RTN per-tensor (symmetric)",
    "int8_rtn_pc": "INT8 RTN per-channel (symmetric)",
    "int4_pc": "INT4 per-channel+KL+AR",
    "quantfuse_int4": "QuantFuse INT4 (mixed+KL+AR)",
}


def _print_table(rep: Dict) -> None:
    agg = rep["aggregate"]
    print("\n" + "=" * 88)
    print(" PTQForge Benchmark — aggregate over", rep["datasets"], "x seeds", rep["seeds"])
    print("=" * 88)
    header = f"{'method':26} {'acc_mean':>9} {'acc_std':>8} {'avg_bits':>9} {'wSQNR_dB':>9}"
    print(header)
    print("-" * 88)
    for key in agg:
        a = agg[key]
        acc = f"{a['acc_mean']:.4f}" if a["acc_mean"] is not None else "  n/a"
        std = f"{a['acc_std']:.4f}" if a["acc_std"] is not None else "  n/a"
        bits = f"{a['avg_bits']:.2f}" if a["avg_bits"] is not None else "  n/a"
        sq = f"{a['sqnr_mean']:.2f}" if a["sqnr_mean"] is not None else "  inf"
        label = _METHOD_LABELS.get(key, key)
        print(f"{label:26} {acc:>9} {std:>8} {bits:>9} {sq:>9}")
    print("-" * 88)
    h = rep.get("headline", {})
    if h.get("available"):
        print(
            f"HEADLINE  QuantFuse={h['quantfuse_acc']:.4f}±{h['quantfuse_std']:.4f}"
            f"  vs INT4-RTN={h['int4_rtn_acc']:.4f}±{h['int4_rtn_std']:.4f}"
            f"  Δ={h['delta_acc']:+.4f}"
            f"  {'SIGNIFICANT' if h['significant_vs_rtn'] else 'not-significant'}"
        )
        if h.get("int8_acc") is not None:
            print(
                f"          vs INT8={h['int8_acc']:.4f}"
                f"  gap={h['gap_to_int8']:+.4f} (mixed-precision cost vs full INT8)"
            )
    print("=" * 88)


def _check_determinism(rep1: Dict, rep2: Dict) -> bool:
    a1 = rep1["aggregate"]
    a2 = rep2["aggregate"]
    ok = True
    for key in a1:
        if a1[key]["accs"] != a2[key]["accs"]:
            ok = False
            print(f"[DETERMINISM] MISMATCH at {key}")
    print(f"[DETERMINISM] {'PASS — bit-for-bit identical' if ok else 'FAIL'}")
    return ok


def _print_synthetic(syn: Dict) -> None:
    ov = syn["overall"]
    h = syn["headline"]
    print("\n" + "=" * 78)
    print(" Synthetic weight-tensor SQNR benchmark (INT4 methods)")
    print("=" * 78)
    header = f"{'method':26} {'SQNR_dB_mean':>14} {'SQNR_dB_std':>13}"
    print(header)
    print("-" * 78)
    labels = {
        "int4_rtn_pt": "INT4 RTN per-tensor",
        "int8_rtn_pc": "INT8 RTN per-channel",
        "int4_pc": "INT4 per-channel+KL+AR",
        "quantfuse_int4": "QuantFuse INT4",
    }
    for m in ["int4_rtn_pt", "int8_rtn_pc", "int4_pc", "quantfuse_int4"]:
        m_, s_ = ov[m]
        print(f"{labels[m]:26} {m_ if m_ is not None else float('nan'):14.2f} {s_ if s_ is not None else 0.0:13.2f}")
    print("-" * 78)
    print(
        f"HEADLINE  QuantFuse INT4 SQNR = {h['quantfuse_sqnr']:.2f} dB  "
        f"vs INT4-RTN {h['int4_rtn_sqnr']:.2f} dB  Δ = +{h['delta_sqnr_db']:.2f} dB  "
        f"(INT8 ceiling {h['int8_sqnr']:.2f} dB)"
    )
    print("=" * 78)


def _print_ablation(abl: Dict) -> None:
    print("\n" + "=" * 88)
    print(" Ablation — QuantFuse component contribution (real acc + weight-SQNR)")
    print("=" * 88)
    print(f" FP32 ceiling acc = {abl['fp32_acc']:.4f}")
    header = f"{'variant':28} {'acc':>8} {'wSQNR_dB':>10} {'avg_bits':>9}"
    print(header)
    print("-" * 88)
    for r in abl["rows"]:
        acc = f"{r['acc']:.4f}"
        sq = f"{r['wsqnr_db']:.2f}" if r.get("wsqnr_db") is not None else "  inf"
        print(f"{r['variant']:28} {acc:>8} {sq:>10} {r['avg_bits']:>9.2f}")
    print("=" * 88)


def _print_failure(fail: Dict) -> None:
    print("\n" + "=" * 88)
    print(" Failure-case analysis (>=3 typical breakages, measured)")
    print("=" * 88)
    for c in fail["cases"]:
        print(f" [{c['id']}] {c['title']}")
        print(
            f"    naive={c['naive_sqnr_db']:.2f} dB  fixed={c['fixed_sqnr_db']:.2f} dB  delta={c['delta_db']:+.2f} dB"
        )
        print(f"    root: {c['root_cause']}")
    print("=" * 88)


def _print_backends(be: Dict) -> None:
    print("\n[backends] offline Tier-1 numpy path always exercised; optional backends:")
    for name, info in be["backends"].items():
        status = info.get("status")
        if status == "skipped":
            print(f"   - {name}: SKIPPED ({info.get('reason')})")
        else:
            print(f"   - {name}: {status}")
    print(f"   offline_fallback = {be['offline_fallback']}")


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - non-Windows
        pass
    ap = argparse.ArgumentParser(prog="ptqforge", description="PTQForge: post-training quantization benchmark (晨星)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--datasets", type=str, nargs="+", default=["moons", "circles", "gaussian"])
    ap.add_argument("--out", type=str, default="benchmark.json")
    ap.add_argument("--no-determinism", action="store_true", help="skip 2nd-run determinism check")
    args = ap.parse_args(argv)

    set_all(1234)
    pipe = QuantPipeline()
    t0 = time.time()
    rep = pipe.benchmark(args.seeds, args.datasets, out_path=args.out)
    elapsed = time.time() - t0
    _print_table(rep)
    print(f"[timing] first run elapsed = {elapsed:.1f}s")

    # Controlled synthetic weight-tensor SQNR benchmark (fast, no training)
    syn = pipe.benchmark_synthetic(args.seeds, out_path="benchmark_synthetic.json")
    _print_synthetic(syn)

    # Ablation: component contribution on a real trained model.
    abl = pipe.ablation_study(dataset_name=args.datasets[0], seed=args.seeds[0], out_path="benchmark_ablation.json")
    _print_ablation(abl)

    # Failure-case analysis (>=3 typical breakages, measured from real runs).
    fail = pipe.analyze_failure_cases(seeds=args.seeds, out_path="benchmark_failure.json")
    _print_failure(fail)

    # Optional SOTA backends (torch / onnxruntime): probe + graceful skip.
    be = benchmark_with_optional_backends(out_path="benchmark_backends.json")
    _print_backends(be)

    det_ok = True
    if not args.no_determinism:
        set_all(1234)
        rep2 = pipe.benchmark(args.seeds, args.datasets)
        det_ok = _check_determinism(rep, rep2)

    # DoD budget guard
    if elapsed > 60:
        print(f"[WARN] demo exceeded 60s budget ({elapsed:.1f}s)")
    return 0 if det_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
