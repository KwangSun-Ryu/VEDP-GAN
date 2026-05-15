"""Deterministic behaviour helpers."""
import os
import random
from typing import Optional

import numpy as np
import torch


def improve_reproducibility(seed: Optional[int]) -> None:
    """Reproduce the seeding strategy provided by libzero.improve_reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # ensure deterministic CuBLAS kernels without requiring manual exports
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
