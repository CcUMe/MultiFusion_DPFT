from __future__ import annotations

from typing import Any

__all__ = ["build_trainer", "train"]


def build_trainer(*args: Any, **kwargs: Any):
    from dprt.training.trainer import build_trainer as _build_trainer

    return _build_trainer(*args, **kwargs)


def train(*args, **kwargs):
    return build_trainer(*args, **kwargs)
