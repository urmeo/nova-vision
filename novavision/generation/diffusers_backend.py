"""Local text-to-image generation with Diffusers."""

from __future__ import annotations

import threading

from PIL import Image

from novavision.config import DIFFUSION_MODEL, DIFFUSION_REVISION, default_revision
from novavision.generation.base import ImageBackend


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DiffusersBackend(ImageBackend):
    """Runs a local diffusers text-to-image pipeline."""

    name = "diffusers"

    def __init__(
        self,
        model_id: str = DIFFUSION_MODEL,
        device: str | None = None,
        steps: int = 2,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or _pick_device()
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self.steps = steps
        self.revision = revision or default_revision(model_id, DIFFUSION_MODEL, DIFFUSION_REVISION)
        self._pipe = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def pipe(self):
        if self._pipe is None:
            with self._load_lock:
                if self._pipe is None:
                    import torch
                    from diffusers import AutoPipelineForText2Image

                    dtype = torch.float16 if self.dtype == "float16" else torch.float32
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        self.model_id, torch_dtype=dtype, revision=self.revision
                    ).to(self.device)
                    pipe.set_progress_bar_config(disable=True)
                    self._pipe = pipe
        return self._pipe

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(int(seed) % (2**63 - 1))
        turbo = "turbo" in self.model_id.lower()

        with self._generate_lock:
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=self.steps,
                guidance_scale=0.0 if turbo else 7.0,
                negative_prompt=None if turbo else negative_prompt,
                generator=generator,
            )
        return result.images[0]
