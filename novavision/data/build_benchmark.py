"Build AffectBench: a balanced Ekman benchmark from GoEmotions."

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from novavision.data import sha256
from novavision.taxonomy import EMOTIONS, to_ekman

DATASET = "google-research-datasets/go_emotions"

DEFAULT_REVISION = "add492243ff905527e67aeb8b80c082af02207c3"


def _normalize(text: str) -> str:
    """Case/whitespace-insensitive key for duplicate detection."""
    return " ".join(text.lower().split())


def _drop_overlap(
    examples: list[tuple[str, str]], exclude_norms: set[str]
) -> list[tuple[str, str]]:
    "Remove examples whose normalised text appears in ``exclude_norms``."
    if not exclude_norms:
        return examples
    return [(text, emotion) for text, emotion in examples if _normalize(text) not in exclude_norms]


def _curate(examples: list[tuple[str, str]], n_per_class: int, seed: int) -> dict[str, list[str]]:
    "Single-label, deduplicated, balanced per-class sampling."
    buckets: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for text, emotion in examples:
        text = text.strip()
        normalized = _normalize(text)
        if not text or normalized in seen or emotion not in EMOTIONS:
            continue
        seen.add(normalized)
        buckets[emotion].append(text)

    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        texts = sorted(buckets.get(emotion, []))
        rng.shuffle(texts)
        sampled[emotion] = texts[:n_per_class]
    return sampled


def _interleave(sampled: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Round-robin across classes so a prefix stays stratified."""
    rows: list[tuple[str, str]] = []
    for index in range(max((len(texts) for texts in sampled.values()), default=0)):
        for emotion in EMOTIONS:
            if index < len(sampled[emotion]):
                rows.append((sampled[emotion][index], emotion))
    return rows


def build(
    n_per_class: int = 100,
    out_path: str | Path = "outputs/generated/affectbench.csv",
    seed: int = 0,
    split: str = "test",
    revision: str = DEFAULT_REVISION,
    drop_train_overlap: bool = True,
) -> Path:
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be a positive integer, got {n_per_class}")

    from datasets import load_dataset

    dataset = load_dataset(DATASET, "simplified", split=split, revision=revision)
    names = dataset.features["labels"].feature.names
    examples = [
        (row["text"], to_ekman(names[row["labels"][0]]))
        for row in dataset
        if len(row["labels"]) == 1
    ]

    before = len(examples)
    if drop_train_overlap and split != "train":
        train = load_dataset(DATASET, "simplified", split="train", revision=revision)
        train_norms = {_normalize(row["text"]) for row in train}
        examples = _drop_overlap(examples, train_norms)
    dropped_overlap = before - len(examples)

    sampled = _curate(examples, n_per_class, seed)
    rows = _interleave(sampled)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["text", "emotion"])
        writer.writerows(rows)

    counts = {emotion: len(sampled[emotion]) for emotion in EMOTIONS}
    underfilled = sorted(emotion for emotion, count in counts.items() if count < n_per_class)
    manifest = {
        "dataset": DATASET,
        "split": split,
        "revision": revision,
        "n_per_class": n_per_class,
        "seed": seed,
        "counts": counts,
        "total": len(rows),
        "underfilled": underfilled,
        "balanced": not underfilled,
        "drop_train_overlap": bool(drop_train_overlap and split != "train"),
        "dropped_train_overlap": dropped_overlap,
        "sha256": sha256(out_path),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if underfilled:
        print(f"WARNING: classes below {n_per_class}: {underfilled}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AffectBench benchmark")
    parser.add_argument("--n", type=int, default=100, help="examples per emotion")
    parser.add_argument("--out", default="outputs/generated/affectbench.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="GoEmotions dataset revision to pin (default: the reproducible pinned commit)",
    )
    parser.add_argument(
        "--keep-train-overlap",
        action="store_true",
        help="do NOT subtract the train split (disables cross-split dedup)",
    )
    args = parser.parse_args()
    path = build(
        args.n,
        args.out,
        args.seed,
        args.split,
        revision=args.revision,
        drop_train_overlap=not args.keep_train_overlap,
    )
    print(f"Wrote {path} and {path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
