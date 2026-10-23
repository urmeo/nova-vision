from __future__ import annotations

from novavision.generation.base import ImageBackend, NullBackend, _require_nonempty_string


def get_backend(name: str = "null", **kwargs) -> ImageBackend:
    _require_nonempty_string(name, "Backend name")
    name = name.lower()
    if name == "null":
        return NullBackend()
    if name == "diffusers":
        from novavision.generation.diffusers_backend import DiffusersBackend

        return DiffusersBackend(**kwargs)
    if name == "hf-api":
        from novavision.generation.hf_api_backend import HFApiBackend

        return HFApiBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'")
