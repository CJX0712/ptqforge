"""Core dataclasses shared across PTQForge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ConfigValidationError


@dataclass
class QuantConfig:
    """Quantization configuration (ENV_XXX_* overrides via core.config)."""

    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    calib_method: str = "minmax"  # minmax | kl
    clip_outliers: bool = False
    kl_bins: int = 2048
    adaptive_rounding: bool = False
    ar_iters: int = 20
    ar_reg: float = 0.01
    ar_gamma_init: float = 2.0
    ar_gamma_end: float = 20.0

    def validate(self) -> "QuantConfig":
        if not (2 <= self.bits <= 16):
            raise ConfigValidationError("bits must be in [2,16]")
        if self.calib_method not in ("minmax", "kl"):
            raise ConfigValidationError(f"unknown calib_method={self.calib_method!r}")
        if self.kl_bins < 8:
            raise ConfigValidationError("kl_bins must be >= 8")
        if self.ar_iters < 0:
            raise ConfigValidationError("ar_iters must be >= 0")
        if self.ar_reg < 0:
            raise ConfigValidationError("ar_reg must be >= 0")
        return self


@dataclass
class QuantResult:
    """One method's evaluation on one (dataset, seed) instance."""

    name: str
    bits: object  # int (uniform) or list[int] (mixed)
    accuracy: float
    sqnr_db: float
    avg_bits: float
    detail: dict = field(default_factory=dict)


@dataclass
class MethodSpec:
    """Registry entry describing a quantization method."""

    key: str
    label: str
    description: str
    is_fp32: bool = False
    mixed: bool = False
