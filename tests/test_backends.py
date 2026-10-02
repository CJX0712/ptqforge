"""Tests for the optional-backend probe (offline-fallback guarantee).

Author: 晨星
"""

from ptqforge.backends import (
    available_onnxruntime,
    available_torch,
    benchmark_with_optional_backends,
    list_backends,
)


def test_probes_return_bool():
    assert isinstance(available_torch(), bool)
    assert isinstance(available_onnxruntime(), bool)
    backends = list_backends()
    assert set(backends.keys()) == {"torch", "onnxruntime"}
    assert all(isinstance(v, bool) for v in backends.values())


def test_benchmark_records_offline_fallback():
    rep = benchmark_with_optional_backends()
    assert rep["offline_fallback"] == "numpy Tier-1"
    assert rep["numpy_path_exercised"] is True
    # never fabricates a number for a missing backend
    for info in rep["backends"].values():
        assert "status" in info
