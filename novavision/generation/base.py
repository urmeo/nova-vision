"""Backend interface and an offline null backend."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class ImageBackend(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image: ...


class NullBackend(ImageBackend):
    """Deterministic synthetic images for offline runs and tests."""

    name = "null"

    def __init__(self, **_):
        pass

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()

        rng = np.random.default_rng((seed ^ int.from_bytes(digest[:8], "big")) & (2**64 - 1))
        tint = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float32)
        noise = rng.integers(0, 64, size=(height, width, 3), dtype=np.uint8)
        pixels = (noise.astype(np.float32) + tint).clip(0, 255).astype(np.uint8)
        return Image.fromarray(pixels)
