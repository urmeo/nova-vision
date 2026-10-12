"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random


def set_determinism(seed: int = 0) -> None:
    "Seed Python, NumPy, and PyTorch, and request deterministic kernels."
    # Python reads this value at startup, so it applies to child processes.
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed % 2**32)

    try:
        import torch

        torch_seed = seed % 2**64
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
