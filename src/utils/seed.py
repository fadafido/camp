"""Global seeding utility for reproducibility.

Call ``set_seed(42)`` at the top of every script and at the top of every
notebook cell that introduces randomness. The fixed seed is a project-wide
convention.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all relevant random number generators for reproducibility.

    Seeds Python's ``random`` module, NumPy, and (if importable) PyTorch on
    both CPU and CUDA. Sets PyTorch's cuDNN backend to deterministic mode.

    Parameters
    ----------
    seed
        The seed value. Defaults to 42 — the project-wide seed used in every
        script in this repository.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
