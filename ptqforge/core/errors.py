"""Structured error taxonomy (E100-E500).

Author: 晨星
"""

from __future__ import annotations


class PtqError(Exception):
    """Base class for all PTQForge errors."""

    code = "E000"

    def __init__(self, msg: str) -> None:
        super().__init__(f"[{self.code}] {msg}")


class ConfigError(PtqError):
    code = "E100"


class ConfigValidationError(ConfigError):
    code = "E101"


class QuantError(PtqError):
    code = "E200"


class DataError(PtqError):
    code = "E300"


class ModelError(PtqError):
    code = "E400"


class PipelineError(PtqError):
    code = "E500"
