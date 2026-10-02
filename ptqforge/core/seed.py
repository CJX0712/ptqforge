"""Global deterministic seeding.

Author: 晨星
"""

from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> int:
    """Seed every RNG used by PTQForge from a single entry point.

    Determinism contract: the same ``seed`` reproduces the full benchmark
    (core metrics bit-for-bit, except wall-clock time). We seed the legacy
    numpy global RNG (stable across runs on the same numpy build) and the
    stdlib ``random``. Downstream code should prefer an explicit
    ``np.random.RandomState(seed)`` for data/model RNG; this helper covers
    any stray global calls.
    """
    seed = int(seed)
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:  # pragma: no cover - defensive
        pass
    return seed
