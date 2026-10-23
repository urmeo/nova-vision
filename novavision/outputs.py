from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def validate_output_paths(outputs: Iterable[str | Path], inputs: Iterable[str | Path]) -> None:
    protected = [Path(path) for path in inputs]
    for path in outputs:
        destination = Path(path)
        for source in protected:
            if destination.resolve() == source.resolve() or (
                destination.exists() and source.exists() and destination.samefile(source)
            ):
                raise ValueError(f"Output path conflicts with an input or another output: {path}")
        protected.append(destination)


@contextmanager
def atomic_output(path: str | Path) -> Iterator[BinaryIO]:
    path = Path(path)
    if path.parent.is_symlink():
        raise ValueError("Output directory cannot be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".nova-", suffix=".tmp", dir=path.parent)
    try:
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                yield stream
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_output(path: str | Path, content: bytes) -> None:
    with atomic_output(path) as stream:
        stream.write(content)


def json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
