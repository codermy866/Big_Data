"""Privacy-aware logging (case_id only in messages)."""

from __future__ import annotations

from src.utils.logging_utils import get_logger as _get

__all__ = ["get_logger"]


def get_logger(name: str):
    return _get(name)
