"""Provenance manifest for a benchmark run."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def package_version(pkg: str) -> str:
    'Installed version of a package, or ``"absent"`` if it is not installed.'
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "absent"


def git_sha() -> str:
    """Current commit SHA, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def device_info() -> dict:
    "Best-effort accelerator provenance."
    info: dict = {
        "cuda_available": False,
        "mps_available": False,
        "device_name": None,
        "cuda": None,
        "cudnn": None,
    }
    try:
        import torch
    except Exception:
        return info
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(torch.backends.mps.is_available())
        info["cuda"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            cudnn = torch.backends.cudnn.version()
            info["cudnn"] = int(cudnn) if cudnn else None
    except Exception:
        pass
    return info


def build_manifest(**config) -> dict:
    """Record model revisions, configuration, and environment details."""
    from novavision.config import (
        CLIP_MODEL,
        CLIP_REVISION,
        DIFFUSION_MODEL,
        DIFFUSION_REVISION,
        EMOTION_MODEL,
        EMOTION_REVISION,
        default_revision,
    )

    def pin(key: str, default_id: str, revision: str) -> str | None:
        revision_key = key.removesuffix("_model") + "_revision"
        if revision_key in config:
            return config[revision_key]
        return default_revision(config.get(key) or default_id, default_id, revision)

    return {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device_info": device_info(),
        "packages": {
            pkg: package_version(pkg)
            for pkg in ("torch", "transformers", "diffusers", "numpy", "datasets", "pillow")
        },
        "model_revisions": {
            "emotion": pin("emotion_model", EMOTION_MODEL, EMOTION_REVISION),
            "diffusion": pin("diffusion_model", DIFFUSION_MODEL, DIFFUSION_REVISION),
            "clip": pin("clip_model", CLIP_MODEL, CLIP_REVISION),
            "probe": config.get("probe_revision", pin("clip_model", CLIP_MODEL, CLIP_REVISION)),
        },
        "config": config,
    }
