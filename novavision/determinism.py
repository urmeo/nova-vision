"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random


def set_determinism(seed: int = 0) -> None:
    "Seed Python, NumPy, and PyTorch, and request deterministic kernels."
    # Python reads this value at startup, so it applies to child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
