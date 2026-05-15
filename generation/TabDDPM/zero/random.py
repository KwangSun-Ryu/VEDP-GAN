"""Randomness helpers mimicking libzero.random."""
from typing import Any, Dict
import random as _random

import numpy as np
import torch


def get_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": _random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_state(state: Dict[str, Any]) -> None:
    _random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_state = state["torch_cuda"]
        if isinstance(cuda_state, (list, tuple)):
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            torch.cuda.set_rng_state(cuda_state)
