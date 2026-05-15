"""Utilities for splitting batches into smaller chunks."""
from collections.abc import Mapping, Sequence
from typing import Any, Iterator

import numpy as np
import torch


def iter_batches(batch: Any, chunk_size: int) -> Iterator[Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = _infer_length(batch)
    if total == 0:
        return
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield _slice(batch, slice(start, end))


def _infer_length(batch: Any) -> int:
    if isinstance(batch, Mapping):
        for value in batch.values():
            return _infer_length(value)
        return 0
    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        if not batch:
            return 0
        return _infer_length(batch[0])
    if hasattr(batch, "__len__"):
        return len(batch)
    raise TypeError("Cannot infer length for the provided batch object")


def _slice(item: Any, sl: slice) -> Any:
    if isinstance(item, Mapping):
        return item.__class__({k: _slice(v, sl) for k, v in item.items()})
    if isinstance(item, list):
        return [_slice(v, sl) for v in item]
    if isinstance(item, tuple):
        return tuple(_slice(v, sl) for v in item)
    if isinstance(item, np.ndarray):
        return item[sl]
    if isinstance(item, torch.Tensor):
        return item[sl]
    try:
        return item[sl]
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError("Batch elements must support slicing") from exc
