"""Configuration loading with ENV_XXX_* overrides and schema validation.

Author: 晨星
"""

from __future__ import annotations

import os

from .errors import ConfigValidationError
from .types import QuantConfig

_PREFIX = "PTQFORGE_"


def _cast_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


_ENV_MAP = {
    "DEFAULT_BITS": ("bits", int),
    "SYMMETRIC": ("symmetric", _cast_bool),
    "PER_CHANNEL": ("per_channel", _cast_bool),
    "CLIP_OUTLIERS": ("clip_outliers", _cast_bool),
    "ADAPTIVE_ROUNDING": ("adaptive_rounding", _cast_bool),
    "CALIB_METHOD": ("calib_method", str),
    "KL_BINS": ("kl_bins", int),
    "AR_ITERS": ("ar_iters", int),
    "AR_REG": ("ar_reg", float),
    "AR_GAMMA_INIT": ("ar_gamma_init", float),
    "AR_GAMMA_END": ("ar_gamma_end", float),
}


def load_config(prefix: str = _PREFIX) -> QuantConfig:
    """Build a QuantConfig from defaults, then apply present ENV overrides."""
    cfg = QuantConfig()
    for env, (attr, cast) in _ENV_MAP.items():
        raw = os.environ.get(prefix + env)
        if raw is None:
            continue
        try:
            setattr(cfg, attr, cast(raw))
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigValidationError(f"invalid {prefix}{env}={raw!r}: {exc}") from exc
    cfg.validate()
    return cfg
