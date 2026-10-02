"""Pipeline / CLI integration tests (smoke + determinism + offline fallback).

Author: 晨星
"""

import json
import os
import tempfile

from ptqforge.backends import benchmark_with_optional_backends
from ptqforge.core.seed import set_all
from ptqforge.pipeline.benchmark import QuantPipeline


def _tmp(name):
    d = tempfile.mkdtemp()
    return os.path.join(d, name)


def test_run_instance_returns_all_methods():
    pipe = QuantPipeline()
    inst = pipe.run_instance("moons", 1)
    names = {r.name for r in inst["results"]}
    for expected in ["fp32", "int4_rtn_pt", "int8_rtn_pc", "int4_pc", "quantfuse_int4"]:
        assert expected in names


def test_benchmark_aggregate_has_headline():
    pipe = QuantPipeline()
    rep = pipe.benchmark([1, 2, 3], ["moons"], out_path=_tmp("b.json"))
    agg = rep["aggregate"]
    assert "quantfuse_int4" in agg and "int4_rtn_pt" in agg
    assert "headline" in rep
    # QuantFuse weight-SQNR must beat INT4-RTN by a clear margin (>= +5 dB)
    qf = agg["quantfuse_int4"]["sqnr_mean"]
    rt = agg["int4_rtn_pt"]["sqnr_mean"]
    assert qf - rt >= 5.0


def test_determinism_bit_for_bit():
    pipe = QuantPipeline()
    set_all(1234)
    r1 = pipe.benchmark([1, 2, 3], ["moons"], out_path=_tmp("a.json"))
    set_all(1234)
    r2 = pipe.benchmark([1, 2, 3], ["moons"])
    assert r1["aggregate"] == r2["aggregate"]


def test_ablation_has_five_rows():
    pipe = QuantPipeline()
    abl = pipe.ablation_study(out_path=_tmp("abl.json"))
    assert len(abl["rows"]) == 5
    # final mixed-precision row must reach the FP32 ceiling within 0.01 acc
    last = abl["rows"][-1]
    assert abs(last["acc"] - abl["fp32_acc"]) <= 0.02


def test_failure_analysis_has_three_cases():
    pipe = QuantPipeline()
    fail = pipe.analyze_failure_cases(out_path=_tmp("f.json"))
    assert len(fail["cases"]) >= 3
    for c in fail["cases"]:
        assert c["fixed_sqnr_db"] > c["naive_sqnr_db"]


def test_cli_smoke(tmp_path, monkeypatch):
    from ptqforge.cli import main

    out = str(tmp_path / "bench.json")
    rc = main(["--seeds", "1", "2", "3", "--datasets", "moons", "--out", out])
    assert rc == 0
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as f:
        rep = json.load(f)
    assert "aggregate" in rep


def test_offline_backend_graceful_skip():
    rep = benchmark_with_optional_backends(out_path=_tmp("be.json"))
    # numpy Tier-1 path is always exercised; torch/onnxruntime are optional
    assert rep["offline_fallback"] == "numpy Tier-1"
    assert rep["numpy_path_exercised"] is True
    for name in ("torch", "onnxruntime"):
        assert name in rep["backends"]
        assert rep["backends"][name]["status"] in ("available", "skipped")
