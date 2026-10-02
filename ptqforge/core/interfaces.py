"""Protocol interfaces for quantizers (architecture contract).

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import QuantConfig


@runtime_checkable
class Quantizer(Protocol):
    """Any object that can quantize a weight tensor under a QuantConfig."""

    def quantize(self, w: np.ndarray, cfg: QuantConfig) -> tuple[np.ndarray, dict]:
        """Return (dequantized_float_tensor, meta_dict)."""
        ...
