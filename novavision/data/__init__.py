"""Benchmark loading, content bank, and provenance checks."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from novavision.taxonomy import EMOTIONS

CONTENT_BANK_PATH = Path(__file__).with_name("content_bank.txt")
LEXICON_PATH = Path(__file__).with_name("affect_lexicon.tsv")

_EMOTIONS = set(EMOTIONS)


def load_benchmark(path: str | Path) -> list[dict[str, str]]:
    "Load validated (text, emotion) rows from a benchmark CSV."
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or {"text", "emotion"} - set(reader.fieldnames):
            raise ValueError(f"{path} must have 'text' and 'emotion' columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path} has duplicate column names")
        for row in reader:
            line_number = reader.line_num
            if None in row:
                raise ValueError(f"{path}:{line_number} has more fields than columns")
            text = (row.get("text") or "").strip()
            emotion = (row.get("emotion") or "").strip().lower()
            if not text:
                raise ValueError(f"{path}:{line_number} has empty text")
            if emotion not in _EMOTIONS:
                raise ValueError(f"{path}:{line_number} unknown emotion '{emotion}'")
            rows.append({"text": text, "emotion": emotion})
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_content_bank(path: str | Path | None = None) -> list[str]:
    "Load neutral, emotion-independent content prompts."
    path = Path(path or CONTENT_BANK_PATH)
    items: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key = " ".join(line.lower().split())
            if key in seen:
                raise ValueError(f"{path}:{line_number} duplicate subject: '{line}'")
            seen.add(key)
            items.append(line)
    if not items:
        raise ValueError(f"{path} has no content prompts")
    return items


def sha256(path: str | Path) -> str:
    """Content hash of a benchmark, for the run manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
