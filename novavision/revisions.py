from __future__ import annotations

import re
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def default_revision(model_id: str, default_model: str, revision: str) -> str | None:
    return revision if model_id == default_model else None


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a nonempty string")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or None")
    model_path = Path(model_id)
    if model_path.is_absolute() or model_path.exists() or model_id.startswith(("./", "../", "~/")):
        raise ValueError("Local model paths cannot be pinned with a Hub revision.")
    if revision is not None and _COMMIT.fullmatch(revision):
        return revision.lower()

    try:
        from huggingface_hub import model_info, try_to_load_from_cache
        from huggingface_hub.utils import RevisionNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "Resolving model revisions requires huggingface-hub; "
            "run python -m pip install -r requirements.txt."
        ) from exc

    requested = "main" if revision is None else revision
    try:
        info = model_info(model_id, revision=requested, timeout=30)
    except RevisionNotFoundError:
        raise
    except Exception as exc:
        for filename in ("config.json", "model_index.json"):
            try:
                cached = try_to_load_from_cache(model_id, filename, revision=requested)
                if not isinstance(cached, str):
                    continue
                path = Path(cached)
                cached_commit = path.parent.name
                if (
                    path.parent.parent.name == "snapshots"
                    and _COMMIT.fullmatch(cached_commit)
                    and path.is_file()
                ):
                    return cached_commit.lower()
            except (OSError, UnicodeError):
                continue
        raise RuntimeError(
            f"Cannot resolve model {model_id!r} revision {requested!r} to an immutable commit. "
            "Connect to the Hub or provide a full commit SHA."
        ) from exc

    commit = info.sha
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError(f"The Hub returned an invalid commit for model {model_id!r}.")
    return commit.lower()
