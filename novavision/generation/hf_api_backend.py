"""HuggingFace Inference API backend (optional, non-deterministic)."""

from __future__ import annotations

import os
import threading

from PIL import Image

from novavision.config import DIFFUSION_MODEL
from novavision.generation.base import ImageBackend


class HFApiBackend(ImageBackend):
    """Generates via the hosted HuggingFace Inference API."""

    name = "hf-api"

    def __init__(self, model_id: str = DIFFUSION_MODEL, token: str | None = None, **_):
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not set")
        self.model_id = model_id
        self._client = None
        self._load_lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            with self._load_lock:
                if self._client is None:
                    from huggingface_hub import InferenceClient

                    self._client = InferenceClient(token=self.token)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        turbo = "turbo" in self.model_id.lower()
        return self.client.text_to_image(
            prompt,
            model=self.model_id,
            width=width,
            height=height,
            negative_prompt=None if turbo else negative_prompt,
            seed=int(seed) % (2**63 - 1),
        )
