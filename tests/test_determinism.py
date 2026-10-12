import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from novavision.determinism import set_determinism


@pytest.mark.parametrize("seed", [-1, 0, 2**32, 2**70])
def test_integer_seeds_repeat_without_torch(seed, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    set_determinism(seed)
    first = (random.random(), np.random.random())
    set_determinism(seed)

    assert (random.random(), np.random.random()) == first
    assert 0 <= int(os.environ["PYTHONHASHSEED"]) < 2**32


def test_torch_seed_stays_within_unsigned_64_bits(monkeypatch):
    seeds = []
    deterministic = []
    torch = SimpleNamespace(
        manual_seed=seeds.append,
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=seeds.append),
        use_deterministic_algorithms=lambda enabled, **kwargs: deterministic.append(enabled),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("PYTHONHASHSEED", "0")

    set_determinism(-1)

    assert seeds == [2**64 - 1, 2**64 - 1]
    assert deterministic == [True]
