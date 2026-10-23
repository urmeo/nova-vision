from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path

from PIL import Image, ImageOps

from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1
from novavision.eval.probes import HFImageClassifierProbe, image_digest
from novavision.outputs import validate_output_paths, write_output
from novavision.revisions import resolve_model_revision
from novavision.taxonomy import EMOTIONS

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


EKMAN_ALIASES = {
    **dict.fromkeys(("happy", "happiness", "amusement", "contentment", "excitement"), "joy"),
    "sad": "sadness",
    "angry": "anger",
    "awe": "surprise",
    **{emotion: emotion for emotion in EMOTIONS},
}


def load_hf_dataset(
    name: str,
    n: int = 200,
    split: str = "train",
    seed: int = 0,
    revision: str | None = None,
    image_key: str = "image",
    label_key: str = "label",
):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("Sample size must be a positive integer")
    import random

    from datasets import load_dataset

    dataset = load_dataset(name, split=split, revision=revision)
    feature = dataset.features.get(label_key)
    names = getattr(feature, "names", None)

    raw_labels = dataset[label_key]
    labels = []
    for label in raw_labels:
        if names and isinstance(label, Integral):
            if isinstance(label, bool) or not -1 <= label < len(names):
                raise ValueError(
                    f"{name}[{split}] column {label_key!r} has invalid class index {label!r}"
                )
            label = names[label] if label >= 0 else None
        labels.append(label)
    emotions = [EKMAN_ALIASES.get(str(label).lower()) for label in labels]

    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    known = set(EMOTIONS)
    pairs: list = []
    for i in order:
        if len(pairs) >= n:
            break
        if emotions[i] in known:
            image = ImageOps.exif_transpose(dataset[i][image_key]).convert("RGB")
            pairs.append((image, emotions[i]))
    if not pairs:
        raise ValueError(
            f"{name}[{split}] column '{label_key}' yielded no Ekman-mappable labels "
            f"(saw e.g. {sorted({str(lab).lower() for lab in labels})[:8]}); pass "
            "--label-key with the column that holds label names "
            "(EmoSet118K: --label-key emotion)"
        )
    return pairs


def load_image_folder(root: str | Path) -> list[tuple[Path, str]]:
    root = Path(root)
    pairs: list[tuple[Path, str]] = []
    known = set(EMOTIONS)
    for folder in sorted(root.iterdir()):
        label = EKMAN_ALIASES.get(folder.name.lower(), folder.name.lower())
        if not folder.is_dir() or label not in known:
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in _EXTS:
                pairs.append((image_path, label))
    if not pairs:
        raise ValueError(f"No labelled images found under {root}/<emotion>/")
    return pairs


def validate(probe, pairs: list) -> dict:
    if not pairs:
        raise ValueError("Probe validation requires at least one labelled image")
    for _, label in pairs:
        if not isinstance(label, str) or label not in EMOTIONS:
            raise ValueError(f"Gold label {label!r} must be one of {EMOTIONS}")
    y_true, y_pred, image_digests = [], [], []
    for item, label in pairs:
        if isinstance(item, Image.Image):
            image = ImageOps.exif_transpose(item).convert("RGB")
        else:
            with Image.open(item) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        image_digests.append(image_digest(image))
        recovery = probe.recover(image)
        y_true.append(label)
        y_pred.append(recovery.emotion)

    cm = confusion_matrix(y_true, y_pred, EMOTIONS)
    return {
        "probe": probe.name,
        "n": len(pairs),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
        "labels": list(EMOTIONS),
        "confusion": cm.tolist(),
        "gold": y_true,
        "predictions": y_pred,
        "image_digests": image_digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a probe on labelled images")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-dir", default=None, help="root/<emotion>/*.jpg")
    source.add_argument("--hf-dataset", default=None, help="labelled HF image dataset")
    parser.add_argument("--n", type=int, default=200, help="sample size for --hf-dataset")

    parser.add_argument("--split", default="train", help="HF dataset split")
    parser.add_argument("--revision", default=None, help="HF dataset revision to pin")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--image-key", default="image", help="HF dataset image column")
    parser.add_argument("--label-key", default="label", help="HF dataset label column")
    parser.add_argument("--probe-model", required=True, help="image-classifier checkpoint")
    parser.add_argument("--probe-revision", default=None, help="probe model commit, branch or tag")
    parser.add_argument("--out", default="outputs/generated/probe_validation.json")
    args = parser.parse_args()

    model = args.probe_model
    requested_revision = args.probe_revision
    revision = resolve_model_revision(model, requested_revision)

    if args.hf_dataset:
        pairs = load_hf_dataset(
            args.hf_dataset,
            n=args.n,
            split=args.split,
            seed=args.seed,
            revision=args.revision,
            image_key=args.image_key,
            label_key=args.label_key,
        )
    else:
        pairs = load_image_folder(args.images_dir)

    validate_output_paths(
        (args.out,), (image for image, _ in pairs if isinstance(image, (str, Path)))
    )
    probe = HFImageClassifierProbe(model_id=model, revision=revision)
    report = validate(probe, pairs)
    report["model"] = model
    report["model_revision"] = revision
    report["requested_model_revision"] = requested_revision
    report["source"] = args.hf_dataset or args.images_dir

    if args.hf_dataset:
        report["split"] = args.split
        report["seed"] = args.seed
        report["revision"] = args.revision
    write_output(args.out, json.dumps(report, indent=2, allow_nan=False).encode("utf-8"))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
