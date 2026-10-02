"""End-to-end demo: trains a reference MLP, quantizes it with every method,
writes benchmark.json, prints the aggregate table, and verifies determinism by
re-running with the same master seed.

Author: 晨星
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ptqforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
