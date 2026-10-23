from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def package_version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "absent"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip().lower()
    if out.returncode == 0 and len(sha) in (40, 64) and all(c in "0123456789abcdef" for c in sha):
        return sha
    return "unknown"


def device_info() -> dict:
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

    def read(getter, default=None):
        try:
            return getter()
        except Exception:
            return default

    info["cuda_available"] = read(lambda: bool(torch.cuda.is_available()), False)
    info["mps_available"] = read(lambda: bool(torch.backends.mps.is_available()), False)
    info["cuda"] = read(lambda: getattr(torch.version, "cuda", None))
    if info["cuda_available"]:
        info["device_name"] = read(lambda: torch.cuda.get_device_name(0))
        cudnn = read(lambda: torch.backends.cudnn.version())
        info["cudnn"] = read(lambda: int(cudnn) if cudnn else None)
    return info


def build_manifest(**config) -> dict:
    from novavision.affect.analyzer import EMOTION_MODEL, EMOTION_REVISION
    from novavision.generation.diffusers_backend import DIFFUSION_MODEL, DIFFUSION_REVISION
    from novavision.revisions import default_revision

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
            "probe": config.get("probe_revision"),
        },
        "config": config,
    }
