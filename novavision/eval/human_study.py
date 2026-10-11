"Human-study harness: sample images for rating, then score agreement."

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from novavision.data import load_content_bank
from novavision.eval.metrics import accuracy, cohen_kappa
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import EMOTIONS


def _sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    """A class-stratified sample from the conditioned tiers."""
    rng = random.Random(seed)
    pool = [record for record in records if record["tier"] in ("emotion", "affect")]
    by_emotion: dict[str, list[dict]] = {emotion: [] for emotion in EMOTIONS}
    for record in pool:
        by_emotion[record["intended"]].append(record)
    for rows in by_emotion.values():
        rng.shuffle(rows)

    picked: list[dict] = []
    limit = min(n, len(pool))
    index = 0
    while len(picked) < limit:
        for emotion in EMOTIONS:
            if index < len(by_emotion[emotion]) and len(picked) < limit:
                picked.append(by_emotion[emotion][index])
        index += 1
    return picked


def _record_index(record: dict, bank: list[str]) -> int:
    "The content/row position used as the seed salt, for reproducing the image."
    if "index" in record:
        return int(record["index"])
    if record["content"] in bank:
        return bank.index(record["content"])
    raise ValueError(
        "Cannot reproduce this image: the record has no 'index' and its content is not a "
        f"content-bank subject ({record['content']!r}). "
        "Rebuild from a run produced by the current code."
    )


def _verify_rebuild_provenance(data: dict) -> None:
    """Reject unverified replay before constructing or loading a model backend."""
    from novavision.experiments.manifest import build_manifest, package_version
    from novavision.experiments.run import _source_fingerprint

    manifest = data["manifest"]
    config = manifest["config"]
    if config["backend"] not in ("diffusers", "null"):
        raise ValueError("Backend is non-deterministic; supply verified saved original images")
    required = ("git_sha", "source_sha256", "python", "platform", "packages", "device_info")
    if any(key not in manifest for key in required):
        raise ValueError("Cannot rebuild images: incomplete provenance; use saved original images")
    current = build_manifest(**config)
    recorded_sha = manifest["git_sha"]
    if (
        not isinstance(recorded_sha, str)
        or len(recorded_sha) != 40
        or any(char not in "0123456789abcdef" for char in recorded_sha)
        or manifest["source_sha256"] != _source_fingerprint()
        or any(manifest[key] != current[key] for key in ("python", "platform", "device_info"))
        or not set(current["packages"]).issubset(manifest["packages"])
        or any(
            package_version(package) != version for package, version in manifest["packages"].items()
        )
    ):
        raise ValueError(
            "Cannot rebuild images: source/environment provenance differs from the run"
        )
    if config["backend"] == "diffusers":
        revision = manifest.get("model_revisions", {}).get("diffusion")
        if not revision or any(
            config.get(key) in (None, "n/a")
            for key in (
                "device",
                "dtype",
                "generation_steps",
                "style",
                "base_seed",
                "width",
                "height",
            )
        ):
            raise ValueError("Cannot rebuild images: generator provenance is incomplete")


def build_sheet(
    results_dir: str | Path, n: int = 60, seed: int = 0, gen=None, *, out: str | Path | None = None
) -> Path:
    "Build blinded ratings from verified images; ``out`` selects the study directory."
    if n < 1:
        raise ValueError("n must be positive")
    results_dir = Path(results_dir)
    data = json.loads((results_dir / "results.json").read_text())
    config = data["manifest"]["config"]
    style = config.get("style", "artistic")
    picked = _sample_records(data["records"], n, seed)
    if not picked:
        raise ValueError("No conditioned records available for a human study")
    if any(not record.get("image_pixel_sha256") for record in picked):
        raise ValueError(
            "Cannot rebuild or verify images: missing pixel digests; use recorded originals"
        )
    rebuild = [record for record in picked if not record.get("image_path")]
    bank = load_content_bank() if rebuild else []

    from PIL import Image

    from novavision.determinism import set_determinism
    from novavision.experiments.run import _image_digest, _seed

    if rebuild:
        _verify_rebuild_provenance(data)

        for record in rebuild:
            _record_index(record, bank)
        set_determinism(config["base_seed"])
        kwargs = {"model_id": config["diffusion_model"]}
        if config["backend"] == "diffusers":
            kwargs.update(
                device=config["device"],
                steps=config["generation_steps"],
                revision=data["manifest"]["model_revisions"]["diffusion"],
            )
        gen = gen or get_backend(config["backend"], **kwargs)
        if config["backend"] == "diffusers" and any(
            getattr(gen, attr, None) != expected
            for attr, expected in (
                ("device", config["device"]),
                ("dtype", config["dtype"]),
                ("steps", config["generation_steps"]),
                ("revision", data["manifest"]["model_revisions"]["diffusion"]),
                ("model_id", config["diffusion_model"]),
            )
        ):
            raise ValueError(
                "Cannot rebuild images: supplied backend differs from recorded generator"
            )
    counts = Counter(record["intended"] for record in picked)
    per_class = {emotion: counts.get(emotion, 0) for emotion in EMOTIONS}

    print(f"[human-study] realized per-class counts: {per_class}", flush=True)

    study = Path(out) if out is not None else results_dir / "human_study"
    images = study / "images"
    images.mkdir(parents=True, exist_ok=True)

    sheet, key = [], []
    for index, record in enumerate(picked):
        if record.get("image_path"):
            source = Path(record["image_path"])
            if not source.is_absolute():
                source = results_dir / source
            with Image.open(source) as saved:
                image = saved.convert("RGB")
        else:
            image = _rebuild_image(record, bank, gen, config, style, _seed)
        if _image_digest(image) != record["image_pixel_sha256"]:
            raise ValueError("Image pixel digest differs from the image the probe scored")
        relative_path = f"images/{index:03d}.png"
        image.save(study / relative_path)
        sheet.append({"id": index, "image": relative_path, "emotion": ""})
        key.append({"id": index, "intended": record["intended"], "probe": record["predicted"]})

    _write_csv(study / "ratings_template.csv", ["id", "image", "emotion"], sheet)
    _write_csv(study / "key.csv", ["id", "intended", "probe"], key)
    _write_csv(
        study / "counts.csv",
        ["emotion", "n"],
        [{"emotion": emotion, "n": count} for emotion, count in per_class.items()],
    )
    return study


def _rebuild_image(record, bank, gen, config, style, seed_fn):
    index = _record_index(record, bank)
    emotion_index = EMOTIONS.index(record["intended"])

    prompt = build_prompt(
        record["content"],
        emotion=record["intended"],
        valence=record["intended_valence"],
        arousal=record["intended_arousal"],
        style=style,
        tier=record["tier"],
    )
    return gen.generate(
        prompt,
        width=config["width"],
        height=config["height"],
        seed=seed_fn(config["base_seed"], index, emotion_index, record["seed"]),
        negative_prompt=NEGATIVE_PROMPT,
    )


def analyze(ratings_csv: str | Path, key_csv: str | Path) -> dict:
    from novavision.eval.validate_probe import EKMAN_ALIASES

    raw = {
        int(record["id"]): record["emotion"].strip().lower() for record in _read_csv(ratings_csv)
    }

    ratings = {
        image_id: EKMAN_ALIASES.get(emotion, emotion)
        for image_id, emotion in raw.items()
        if emotion
    }
    key = {int(record["id"]): record for record in _read_csv(key_csv)}
    known = set(EMOTIONS)

    ids = [image_id for image_id in key if ratings.get(image_id) in known]
    unscored = sorted(
        image_id for image_id in key if ratings.get(image_id) and ratings[image_id] not in known
    )
    human = [ratings[image_id] for image_id in ids]
    probe = [key[image_id]["probe"] for image_id in ids]
    intended = [key[image_id]["intended"] for image_id in ids]
    return {
        "n_rated": len(ids),
        "n_unscored": len(unscored),
        "unscored_ids": unscored,
        "human_vs_probe_kappa": round(cohen_kappa(human, probe, EMOTIONS), 4),
        "human_vs_intended_acc": round(accuracy(human, intended), 4),
        "probe_vs_intended_acc": round(accuracy(probe, intended), 4),
    }


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-study harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--results", default="outputs/results")
    build_parser.add_argument("--n", type=int, default=60)
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument(
        "--out",
        default="outputs/generated/human_study",
        help="destination directory for ratings, images and counts",
    )

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--ratings", required=True)
    analyze_parser.add_argument("--key", required=True)

    args = parser.parse_args()
    if args.cmd == "build":
        path = build_sheet(args.results, n=args.n, seed=args.seed, out=args.out)
        print(f"Wrote rating sheet to {path}")
    else:
        from novavision.experiments.run import json_safe

        print(json.dumps(json_safe(analyze(args.ratings, args.key)), indent=2))


if __name__ == "__main__":
    main()
