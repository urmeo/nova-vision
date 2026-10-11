"Validate a recovery probe against a labelled image-emotion dataset."

from __future__ import annotations

import argparse
import json
from pathlib import Path

from novavision.config import CLIP_MODEL, CLIP_REVISION, default_revision
from novavision.eval.metrics import accuracy, confusion_matrix, macro_f1
from novavision.taxonomy import EMOTIONS

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


EKMAN_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "contentment": "joy",
    "excitement": "joy",
    "joy": "joy",
    "sad": "sadness",
    "sadness": "sadness",
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "surprise": "surprise",
    "awe": "surprise",
    "neutral": "neutral",
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
    "Sample (image, ekman_label) pairs from a labelled HF image dataset."
    import random

    from datasets import load_dataset

    dataset = load_dataset(name, split=split, revision=revision)
    feature = dataset.features.get(label_key)
    names = getattr(feature, "names", None)

    raw_labels = dataset[label_key]
    labels = [names[label] if names and isinstance(label, int) else label for label in raw_labels]
    emotions = [EKMAN_ALIASES.get(str(label).lower()) for label in labels]

    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    known = set(EMOTIONS)
    pairs: list = []
    for i in order:
        if len(pairs) >= n:
            break
        if emotions[i] in known:
            pairs.append((dataset[i][image_key].convert("RGB"), emotions[i]))
    if not pairs:
        raise ValueError(
            f"{name}[{split}] column '{label_key}' yielded no Ekman-mappable labels "
            f"(saw e.g. {sorted({str(lab).lower() for lab in labels})[:8]}); pass "
            "--label-key with the column that holds label names "
            "(EmoSet118K: --label-key emotion)"
        )
    return pairs


def load_image_folder(root: str | Path) -> list[tuple[Path, str]]:
    """Images under ``root/<ekman_emotion>/*`` → (path, label) pairs."""
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
    """pairs are (image-or-path, ekman_label)."""
    from PIL import Image

    y_true, y_pred = [], []
    for item, label in pairs:
        if isinstance(item, Image.Image):
            image = item
        else:
            with Image.open(item) as source:
                image = source.convert("RGB")
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
    }


def clip_revision_for(args) -> str | None:
    "The pinned CLIP revision applies only to the model it pins."
    if args.clip_revision:
        return args.clip_revision
    return default_revision(args.clip_model, CLIP_MODEL, CLIP_REVISION)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a probe on labelled images")
    parser.add_argument("--images-dir", default=None, help="root/<emotion>/*.jpg")
    parser.add_argument("--hf-dataset", default=None, help="labelled HF image dataset")
    parser.add_argument("--n", type=int, default=200, help="sample size for --hf-dataset")

    parser.add_argument("--split", default="train", help="HF dataset split")
    parser.add_argument("--revision", default=None, help="HF dataset revision to pin")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument("--image-key", default="image", help="HF dataset image column")
    parser.add_argument("--label-key", default="label", help="HF dataset label column")
    parser.add_argument("--probe", default="clip", choices=["clip", "hf"])
    parser.add_argument("--clip-model", default=CLIP_MODEL)
    parser.add_argument(
        "--probe-model",
        default=None,
        help="image-classifier model id, required for --probe hf",
    )
    parser.add_argument(
        "--clip-revision",
        default=None,
        help="model revision; defaults to the pinned revision for the default model only",
    )
    parser.add_argument("--out", default="outputs/generated/probe_validation.json")
    args = parser.parse_args()

    from novavision.eval.probes import CLIPProbe, HFImageClassifierProbe

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
    elif args.images_dir:
        pairs = load_image_folder(args.images_dir)
    else:
        parser.error("pass --hf-dataset or --images-dir")

    from novavision.eval.probes import Probe

    probe: Probe
    if args.probe == "hf":
        if not args.probe_model:
            parser.error("--probe hf requires --probe-model")
        probe = HFImageClassifierProbe(model_id=args.probe_model)
        model = args.probe_model
    else:
        model = args.clip_model
        probe = CLIPProbe(model_id=model, revision=clip_revision_for(args))
    report = validate(probe, pairs)
    report["model"] = model
    report["source"] = args.hf_dataset or args.images_dir

    if args.hf_dataset:
        report["split"] = args.split
        report["seed"] = args.seed
        report["revision"] = args.revision
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
