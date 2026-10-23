from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from numbers import Integral

import numpy as np
from PIL import Image


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_request(
    prompt: str, width: int, height: int, seed: int, negative_prompt: str | None
) -> tuple[int, int, int]:
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if negative_prompt is not None and not isinstance(negative_prompt, str):
        raise ValueError("negative_prompt must be a string or None")
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    return int(width), int(height), int(seed)


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

    name = "null"

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        width, height, seed = _validate_request(prompt, width, height, seed, negative_prompt)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()

        rng = np.random.default_rng((seed ^ int.from_bytes(digest[:8], "big")) & (2**64 - 1))
        tint = np.frombuffer(digest[:3], dtype=np.uint8)
        noise = rng.integers(0, 64, size=(height, width, 3), dtype=np.uint8)
        pixels = np.minimum(np.add(noise, tint, dtype=np.uint16), 255).astype(np.uint8)
        return Image.fromarray(pixels)
