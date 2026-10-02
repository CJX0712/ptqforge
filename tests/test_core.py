"""Tests for core utilities (config, seed, errors, types).

Author: 晨星
"""

import numpy as np
import pytest

from ptqforge.core.config import load_config
from ptqforge.core.errors import ConfigValidationError, PtqError
from ptqforge.core.seed import set_all
from ptqforge.core.types import QuantConfig


def test_config_default():
    cfg = load_config()
    assert isinstance(cfg, QuantConfig)
    assert cfg.bits == 8


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("PTQFORGE_DEFAULT_BITS", "4")
    monkeypatch.setenv("PTQFORGE_CLIP_OUTLIERS", "true")
    monkeypatch.setenv("PTQFORGE_ADAPTIVE_ROUNDING", "1")
    cfg = load_config()
    assert cfg.bits == 4
    assert cfg.clip_outliers is True
    assert cfg.adaptive_rounding is True


def test_config_validate_rejects_bad_bits():
    with pytest.raises(ConfigValidationError):
        QuantConfig(bits=1).validate()
    with pytest.raises(ConfigValidationError):
        QuantConfig(calib_method="nonsense").validate()


def test_seed_returns_int_and_is_deterministic():
    s = set_all(7)
    assert s == 7
    a = np.random.RandomState(3).randn(5)
    b = np.random.RandomState(3).randn(5)
    assert np.allclose(a, b)


def test_error_hierarchy():
    assert issubclass(ConfigValidationError, PtqError)
    e = ConfigValidationError("boom")
    assert "[E101]" in str(e)
