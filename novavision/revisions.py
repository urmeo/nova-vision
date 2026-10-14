"""Resolve Hub model references to immutable commits."""

from __future__ import annotations

import re
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    """Resolve a branch or tag online, falling back to its cached commit offline."""
    model_path = Path(model_id)
    if model_path.is_absolute() or model_path.exists() or model_id.startswith(("./", "../", "~/")):
        raise ValueError("Local model paths cannot be pinned with a Hub revision.")
    if revision and _COMMIT.fullmatch(revision):
        return revision.lower()

    try:
        from huggingface_hub import model_info, try_to_load_from_cache
    except ImportError as exc:
        raise RuntimeError(
            "Resolving model revisions requires huggingface-hub; install .[ml]."
        ) from exc

    requested = revision or "main"
    try:
        info = model_info(model_id, revision=requested, timeout=30)
    except Exception as exc:
        for filename in ("config.json", "model_index.json"):
            cached = try_to_load_from_cache(model_id, filename, revision=requested)
            if isinstance(cached, str):
                path = Path(cached)
                cached_commit = path.parent.name
                if (
                    path.parent.parent.name == "snapshots"
                    and _COMMIT.fullmatch(cached_commit)
                    and path.is_file()
                ):
                    return cached_commit.lower()
        raise RuntimeError(
            f"Cannot resolve model {model_id!r} revision {requested!r} to an immutable commit. "
            "Connect to the Hub or provide a full commit SHA."
        ) from exc

    commit = info.sha
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"The Hub returned an invalid commit for model {model_id!r}.")
    return commit.lower()
