"""HuggingFace Inference API backend (optional, non-deterministic)."""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from novavision.config import DEFAULT_HF_TIMEOUT, DIFFUSION_MODEL
from novavision.generation.base import ImageBackend


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    for module_name, exception_name in (("httpx", "TimeoutException"), ("requests", "Timeout")):
        try:
            module = __import__(module_name, fromlist=[exception_name])
        except ImportError:
            continue
        if isinstance(error, getattr(module, exception_name)):
            return True
    return False


def _generate_image(connection, image_path, model_id, token, timeout, prompt, parameters):
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=token, timeout=timeout)
        image = client.text_to_image(prompt, model=model_id, **parameters)
        image.save(image_path, format="PNG")
        connection.send(("ok", ""))
    except Exception as error:
        status = "timeout" if _is_timeout(error) else "error"
        connection.send((status, f"{type(error).__name__}: {error}"[:512]))
    finally:
        connection.close()


class HFApiBackend(ImageBackend):
    name = "hf-api"

    def __init__(
        self,
        model_id: str = DIFFUSION_MODEL,
        token: str | None = None,
        timeout: float = DEFAULT_HF_TIMEOUT,
        **_,
    ):
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Hosted generation timeout must be finite and greater than zero.")
        self.timeout = float(timeout)
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not set")
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        parameters = {
            "width": width,
            "height": height,
            "negative_prompt": None if "turbo" in self.model_id.lower() else negative_prompt,
            "seed": int(seed) % (2**63 - 1),
        }
        context = multiprocessing.get_context("spawn")
        deadline = time.monotonic() + self.timeout
        with TemporaryDirectory(prefix="novavision-hf-") as directory:
            receiver, sender = context.Pipe(duplex=False)
            image_path = Path(directory) / "image.png"
            worker = context.Process(
                target=_generate_image,
                args=(
                    sender,
                    image_path,
                    self.model_id,
                    self.token,
                    self.timeout,
                    prompt,
                    parameters,
                ),
                daemon=False,
            )
            started = False
            try:
                worker.start()
                started = True
                sender.close()
                if not receiver.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError(
                        f"Hosted generation timed out after {self.timeout:g} seconds."
                    )
                try:
                    status, message = receiver.recv()
                except EOFError:
                    raise RuntimeError(
                        "Hosted generation worker exited without a result."
                    ) from None
                if status == "timeout":
                    raise TimeoutError(message)
                if status != "ok":
                    raise RuntimeError(f"Hosted generation failed: {message}")
                worker.join(max(0, deadline - time.monotonic()))
                if worker.is_alive():
                    raise TimeoutError(
                        f"Hosted generation timed out after {self.timeout:g} seconds."
                    )
                with Image.open(image_path) as image:
                    return image.copy()
            finally:
                if started:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=1)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=1)
                    else:
                        worker.join()
                    if not worker.is_alive():
                        worker.close()
                sender.close()
                receiver.close()
