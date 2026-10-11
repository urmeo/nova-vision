"Affect-recovery benchmark across conditioning tiers, with floors and stats."

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.config import CLIP_MODEL, CLIP_REVISION, default_revision
from novavision.data import (
    CONTENT_BANK_PATH,
    LEXICON_PATH,
    load_benchmark,
    load_content_bank,
    sha256,
)
from novavision.determinism import set_determinism
from novavision.eval import figures
from novavision.eval.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_corr_ci,
    confusion_matrix,
    macro_f1,
    mae,
    majority_baseline,
    paired_bootstrap_test,
    pearson,
    permutation_test,
    prediction_collapse,
    spearman,
)
from novavision.eval.probes import CLIPProbe, HFImageClassifierProbe
from novavision.experiments.manifest import build_manifest, package_version
from novavision.generation import get_backend
from novavision.prompting import NEGATIVE_PROMPT, TIERS, build_prompt
from novavision.taxonomy import EMOTIONS, prior

CONDITIONS = {
    "content": (*TIERS, "scene"),
    "text": (*TIERS, "shuffled"),
}


ALL_CONDITIONS = (*TIERS, "scene", "shuffled")
CONTRASTS = (
    ("naive", "raw"),
    ("emotion", "naive"),
    ("emotion", "raw"),
    ("affect", "emotion"),
    ("affect", "raw"),
)


SEED_MAX_ITEMS = 1030
SEED_MAX_SEEDS = 13


def _seed(base: int, item_index: int, emotion_index: int, seed_index: int) -> int:
    "Shared per (content, emotion, seed) so tiers are paired on noise."
    return base + ((item_index * 97 + emotion_index * 13 + seed_index) % 100_000)


def _check_seed_domain(n_items: int, seeds: int) -> None:
    """Refuse configurations where _seed would stop being collision-free."""
    if n_items > SEED_MAX_ITEMS or seeds > SEED_MAX_SEEDS:
        raise ValueError(
            f"seed mixing is collision-free only up to {SEED_MAX_ITEMS} items and "
            f"{SEED_MAX_SEEDS} seeds (got {n_items} items, {seeds} seeds); "
            "shard the run over --base-seed values at least 100000 apart instead"
        )


def _shuffle_emotion(gold: str, seed: int) -> str:
    """A deterministic wrong emotion, for the shuffled floor."""
    others = [e for e in EMOTIONS if e != gold]
    return others[seed % len(others)]


class _Checkpoint:
    "Stream records to JSONL so a long run resumes instead of restarting."

    def __init__(self, path: Path | None, manifest: dict, *, images_dir: Path | None = None):
        self.path = path
        self.images_dir = images_dir
        self.done: dict = {}
        if path and path.exists():
            lines = path.read_text().splitlines()
            try:
                matches = bool(lines) and json.loads(lines[0]) == {"manifest": manifest}
            except json.JSONDecodeError:
                matches = False
            if not matches:
                raise ValueError(
                    "Resume checkpoint does not match this run's provenance; "
                    "use a fresh output directory."
                )
            for line in lines[1:]:
                if line.strip():
                    r = json.loads(line)
                    key = (r["tier"], r["intended"], r["index"], r["seed"])
                    if key in self.done and self.done[key] != r:
                        raise ValueError("Resume checkpoint contains conflicting records")
                    self.done[key] = r
            if images_dir is not None:
                for r in self.done.values():
                    self._verify_original(r)
        elif path:
            with path.open("x", encoding="utf-8") as fh:
                fh.write(json.dumps({"manifest": manifest}, allow_nan=False) + "\n")

    def cached(self, tier: str, intended: str, index: int, seed: int):
        return self.done.get((tier, intended, index, seed))

    def _verify_original(self, rec: dict) -> None:
        from PIL import Image

        assert self.images_dir is not None
        relative = rec.get("image_path")
        digest = rec.get("image_pixel_sha256")
        if not relative or not digest:
            raise ValueError("Resume checkpoint is missing a saved original image reference/digest")
        source = (self.images_dir.parent / relative).resolve()
        if self.images_dir.resolve() not in source.parents or not source.is_file():
            raise ValueError("Saved original image is missing from the resumed run")
        try:
            with Image.open(source) as image:
                actual_digest = _image_digest(image)
        except OSError as exc:
            raise ValueError("Saved original image could not be decoded") from exc
        if actual_digest != digest:
            raise ValueError("Saved original image pixel digest differs from the checkpoint")

    def record(self, rec: dict, *, image=None) -> dict:
        if self.images_dir is not None:
            if image is None or _image_digest(image) != rec.get("image_pixel_sha256"):
                raise ValueError("Saved original must match the pixels recorded by the probe")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            name = f"{rec['tier']}-{rec['index']:05d}-{rec['intended']}-{rec['seed']:03d}.png"
            target = self.images_dir / name
            temporary = target.with_suffix(".tmp")
            image.save(temporary, format="PNG")
            temporary.replace(target)
            rec = {**rec, "image_path": f"images/{name}"}
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(json_safe(rec)) + "\n")
        return rec


def _source_fingerprint() -> str:
    """Hash the package source, including uncommitted changes, in stable path order."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _image_digest(image) -> str:
    """Digest decoded RGB pixels plus dimensions; this is not a file-byte digest."""
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}:{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def _make_probe(kind: str, probe_model: str | None, clip_model: str, device: str | None):
    """CLIP (default) or an independent HF image-emotion classifier."""
    if kind == "hf":
        if not probe_model:
            raise ValueError("--probe hf requires --probe-model")
        return HFImageClassifierProbe(model_id=probe_model, device=device)
    if kind != "clip":
        raise ValueError(f"unknown probe kind '{kind}', expected 'clip' or 'hf'")
    model_id = probe_model or clip_model
    revision = default_revision(model_id, CLIP_MODEL, CLIP_REVISION)
    return CLIPProbe(model_id=model_id, device=device, revision=revision)


def run_experiment(
    backend: str = "diffusers",
    *,
    track: str = "content",
    benchmark: str | None = None,
    contents: int | None = None,
    limit: int | None = None,
    seeds: int = 3,
    base_seed: int = 0,
    style: str = "artistic",
    out: str = "outputs/generated/run",
    width: int = 512,
    height: int = 512,
    device: str | None = None,
    diffusion_model: str = "stabilityai/sd-turbo",
    probe: str = "clip",
    clip_model: str = CLIP_MODEL,
    probe_model: str | None = None,
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    coverage_override: float | None = None,
    resume: bool = False,
    save_images: bool = False,
) -> dict:
    if track not in CONDITIONS:
        raise ValueError(f"Unknown track: {track}")
    if seeds < 1 or width < 1 or height < 1:
        raise ValueError("Seeds, width and height must be positive")
    if (contents is not None and contents < 0) or (limit is not None and limit < 0):
        raise ValueError("Contents and limit cannot be negative")
    if track == "text":
        if not benchmark:
            raise ValueError("The text track requires --benchmark.")
        rows = load_benchmark(benchmark)
        if limit is not None:
            rows = rows[:limit]
        n_items = len(rows)
    else:
        bank = load_content_bank()
        if contents is not None:
            bank = bank[:contents]
        n_items = len(bank)

    _check_seed_domain(n_items, seeds)
    set_determinism(base_seed)
    kwargs = {"model_id": diffusion_model}
    if device:
        kwargs["device"] = device
    gen = get_backend(backend, **kwargs)
    probe_obj = _make_probe(probe, probe_model, clip_model, device)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lexicon_path = Path(os.getenv("NOVAVISION_LEXICON") or LEXICON_PATH)
    manifest = build_manifest(
        backend=backend,
        track=track,
        diffusion_model=diffusion_model,
        probe=probe_obj.name,
        clip_model=probe_model or clip_model,
        emotion_model=emotion_model if track == "text" else None,
        requested_device=device,
        device=getattr(gen, "device", "n/a"),
        dtype=getattr(gen, "dtype", "n/a"),
        generation_steps=getattr(gen, "steps", None),
        style=style,
        items=n_items,
        seeds=seeds,
        base_seed=base_seed,
        width=width,
        height=height,
        benchmark=benchmark,
        benchmark_sha256=sha256(benchmark) if benchmark else None,
        content_bank_sha256=sha256(CONTENT_BANK_PATH) if track == "content" else None,
        sampled_data_sha256=hashlib.sha256(
            json.dumps(rows if track == "text" else bank, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        lexicon_sha256=sha256(lexicon_path) if track == "text" else None,
        coverage_override=coverage_override,
        save_images=save_images,
    )
    manifest["source_sha256"] = _source_fingerprint()
    manifest["packages"].update(
        {pkg: package_version(pkg) for pkg in ("accelerate", "huggingface-hub", "safetensors")}
    )

    manifest["model_revisions"]["diffusion"] = getattr(gen, "revision", None)
    ckpt = _Checkpoint(
        out_dir / "records.jsonl" if resume else None,
        manifest,
        images_dir=out_dir / "images" if save_images else None,
    )
    if track == "text":
        analyzer = EmotionAnalyzer(model_name=emotion_model, coverage_override=coverage_override)
        records = _text_records(
            rows, gen, probe_obj, analyzer, style, seeds, base_seed, width, height, ckpt
        )
    else:
        records = _content_records(
            bank, gen, probe_obj, style, seeds, base_seed, width, height, ckpt
        )
    conditions = CONDITIONS[track]
    metrics = _summarize(records, conditions)
    contrasts = _contrasts(records)
    _write(out, records, metrics, contrasts, manifest, conditions)
    if ckpt.path and ckpt.path.exists():
        ckpt.path.unlink()
    return {"metrics": metrics, "contrasts": contrasts}


def _content_records(bank, gen, probe, style, seeds, base_seed, width, height, ckpt) -> list[dict]:
    records: list[dict] = []
    total = len(bank) * len(EMOTIONS) * seeds
    step = 0
    for content_index, content in enumerate(bank):
        for emotion_index, emotion in enumerate(EMOTIONS):
            valence, arousal = prior(emotion)
            for seed_index in range(seeds):
                seed = _seed(base_seed, content_index, emotion_index, seed_index)
                step += 1
                print(f"[{step}/{total}] {emotion} :: {content[:30]}", flush=True)
                for tier in TIERS:
                    done = ckpt.cached(tier, emotion, content_index, seed_index)
                    if done is not None:
                        records.append(done)
                        continue
                    image = _render(
                        gen, content, emotion, valence, arousal, style, tier, seed, width, height
                    )
                    recovery = probe.recover(image)
                    clip_t = probe.clip_t(image, content)
                    record = _record(
                        probe.name,
                        tier,
                        content,
                        emotion,
                        seed_index,
                        recovery,
                        valence,
                        arousal,
                        clip_t,
                        index=content_index,
                        image_pixel_sha256=_image_digest(image),
                    )
                    records.append(ckpt.record(record, image=image))

    for emotion_index, emotion in enumerate(EMOTIONS):
        valence, arousal = prior(emotion)
        for seed_index in range(seeds):
            done = ckpt.cached("scene", emotion, 0, seed_index)
            if done is not None:
                records.append(done)
                continue
            seed = _seed(base_seed, 0, emotion_index, seed_index)
            image = _render(gen, "", emotion, valence, arousal, style, "scene", seed, width, height)
            recovery = probe.recover(image)
            record = _record(
                probe.name,
                "scene",
                "",
                emotion,
                seed_index,
                recovery,
                valence,
                arousal,
                float("nan"),
                image_pixel_sha256=_image_digest(image),
            )
            records.append(ckpt.record(record, image=image))
    return records


def _text_records(rows, gen, probe, analyzer, style, seeds, base_seed, width, height, ckpt):
    records: list[dict] = []
    total = len(rows) * seeds
    step = 0
    for row_index, row in enumerate(rows):
        text, gold = row["text"], row["emotion"]
        analysis = analyzer.analyze(text)
        valence, arousal = analysis.valence, analysis.arousal
        emotion_index = EMOTIONS.index(gold)
        for seed_index in range(seeds):
            step += 1
            print(f"[{step}/{total}] {gold} :: {text[:30]}", flush=True)
            seed = _seed(base_seed, row_index, emotion_index, seed_index)
            for tier in TIERS:
                done = ckpt.cached(tier, gold, row_index, seed_index)
                if done is not None:
                    records.append(done)
                    continue
                image = _render(gen, text, gold, valence, arousal, style, tier, seed, width, height)
                recovery = probe.recover(image)
                clip_t = probe.clip_t(image, text)
                record = _record(
                    probe.name,
                    tier,
                    text,
                    gold,
                    seed_index,
                    recovery,
                    valence,
                    arousal,
                    clip_t,
                    classified=analysis.primary,
                    index=row_index,
                    image_pixel_sha256=_image_digest(image),
                )
                records.append(ckpt.record(record, image=image))

            wrong = _shuffle_emotion(gold, seed)
            done = ckpt.cached("shuffled", wrong, row_index, seed_index)
            if done is not None:
                records.append(done)
                continue
            wrong_valence, wrong_arousal = prior(wrong)
            wrong_seed = _seed(base_seed, row_index, EMOTIONS.index(wrong), seed_index)
            image = _render(
                gen,
                text,
                wrong,
                wrong_valence,
                wrong_arousal,
                style,
                "emotion",
                wrong_seed,
                width,
                height,
            )
            recovery = probe.recover(image)
            clip_t = probe.clip_t(image, text)
            record = _record(
                probe.name,
                "shuffled",
                text,
                wrong,
                seed_index,
                recovery,
                wrong_valence,
                wrong_arousal,
                clip_t,
                classified=analysis.primary,
                index=row_index,
                image_pixel_sha256=_image_digest(image),
            )
            records.append(ckpt.record(record, image=image))
    return records


def _render(gen, content, emotion, valence, arousal, style, tier, seed, width, height):
    prompt = build_prompt(
        content, emotion=emotion, valence=valence, arousal=arousal, style=style, tier=tier
    )
    return gen.generate(
        prompt, width=width, height=height, seed=seed, negative_prompt=NEGATIVE_PROMPT
    )


def _record(
    probe,
    tier,
    content,
    emotion,
    seed_index,
    recovery,
    valence,
    arousal,
    clip_t,
    *,
    classified=None,
    index=0,
    image_pixel_sha256,
) -> dict:
    return {
        "probe": probe,
        "tier": tier,
        "content": content,
        "index": index,
        "intended": emotion,
        "classified": classified,
        "seed": seed_index,
        "predicted": recovery.emotion,
        "intended_valence": valence,
        "intended_arousal": arousal,
        "recovered_valence": recovery.valence,
        "recovered_arousal": recovery.arousal,
        "clip_t": clip_t,
        "image_pixel_sha256": image_pixel_sha256,
    }


def _summarize(records, conditions) -> dict:
    summary: dict = {}
    for tier in conditions:
        tier_records = [record for record in records if record["tier"] == tier]
        if not tier_records:
            continue
        y_true = [record["intended"] for record in tier_records]
        y_pred = [record["predicted"] for record in tier_records]
        correct = [int(t == p) for t, p in zip(y_true, y_pred)]
        lo, hi = bootstrap_ci(correct)
        intended_valence, recovered_valence = (
            _col(tier_records, "intended_valence"),
            _col(tier_records, "recovered_valence"),
        )
        intended_arousal, recovered_arousal = (
            _col(tier_records, "intended_arousal"),
            _col(tier_records, "recovered_arousal"),
        )

        clip = [record["clip_t"] for record in tier_records if _finite(record["clip_t"])]
        valence_low, valence_high = bootstrap_corr_ci(intended_valence, recovered_valence)
        arousal_low, arousal_high = bootstrap_corr_ci(intended_arousal, recovered_arousal)
        summary[tier] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "accuracy_ci": [round(lo, 4), round(hi, 4)],
            "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
            "valence_r": round(pearson(intended_valence, recovered_valence), 4),
            "valence_rho": round(spearman(intended_valence, recovered_valence), 4),
            "valence_rho_ci": [round(valence_low, 4), round(valence_high, 4)],
            "valence_mae": round(mae(intended_valence, recovered_valence), 4),
            "arousal_r": round(pearson(intended_arousal, recovered_arousal), 4),
            "arousal_rho": round(spearman(intended_arousal, recovered_arousal), 4),
            "arousal_rho_ci": [round(arousal_low, 4), round(arousal_high, 4)],
            "arousal_mae": round(mae(intended_arousal, recovered_arousal), 4),
            "clip_t": round(sum(clip) / len(clip), 4) if clip else float("nan"),
            "majority_baseline": round(majority_baseline(y_true), 4),
            "collapse": _round_collapse(prediction_collapse(y_pred)),
            "shuffled_control": permutation_test(y_true, y_pred),
            "n": len(tier_records),
        }
    summary["chance"] = round(1 / len(EMOTIONS), 4)
    summary["probe_health"] = _probe_health(records)
    classification_accuracy = _classification_accuracy(records)
    if classification_accuracy is not None:
        summary["classification_accuracy"] = classification_accuracy
    return summary


def _round_collapse(collapse: dict) -> dict:
    return {
        "label": collapse["label"],
        "rate": round(float(collapse["rate"]), 4),
        "distinct": collapse["distinct"],
    }


def _probe_health(records) -> dict:
    "Probe-degeneracy diagnostic over the conditioning tiers (floors excluded)."
    predictions = [record["predicted"] for record in records if record["tier"] in TIERS]
    collapse = prediction_collapse(predictions)
    return {
        "majority_label": collapse["label"],
        "majority_rate": round(float(collapse["rate"]), 4),
        "distinct_labels": collapse["distinct"],
        "n_labels": len(EMOTIONS),
    }


def _classification_accuracy(records) -> float | None:
    """Upstream classifier vs gold, once per input (text track only)."""
    seen, intended, classified = set(), [], []
    for record in records:
        item = record.get("index", record["content"])
        if record.get("classified") is None or item in seen:
            continue
        seen.add(item)
        intended.append(record["intended"])
        classified.append(record["classified"])
    return round(accuracy(intended, classified), 4) if intended else None


def _contrasts(records) -> dict:
    """Paired bootstrap on per-item recovery correctness between tiers."""

    by_key: dict[tuple, dict[str, int]] = {}
    for record in records:
        key = (record.get("index", record["content"]), record["intended"], record["seed"])
        by_key.setdefault(key, {})[record["tier"]] = int(record["intended"] == record["predicted"])

    contrasts = {}
    for higher, lower in CONTRASTS:
        higher_scores, lower_scores = [], []
        for tiers in by_key.values():
            if higher in tiers and lower in tiers:
                higher_scores.append(tiers[higher])
                lower_scores.append(tiers[lower])
        if higher_scores:
            result = paired_bootstrap_test(higher_scores, lower_scores)
            contrasts[f"{higher}_vs_{lower}"] = {k: round(v, 4) for k, v in result.items()}
    return contrasts


def _col(rows, key):
    return [r[key] for r in rows]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def json_safe(obj):
    """Replace non-finite floats (nan/inf) with null so output is valid RFC-8259 JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def dump_results(payload: dict, path) -> None:
    """Write results.json as standards-compliant JSON (no bare NaN tokens)."""
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))


def _write(out, records, metrics, contrasts, manifest, conditions) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "metrics": metrics,
        "contrasts": contrasts,
        "records": records,
    }
    dump_results(payload, out_dir / "results.json")
    _write_figures(out_dir, records, metrics, conditions)


def _write_figures(out_dir, records, metrics, conditions, *, figures_dir=None) -> None:
    "Render the accuracy, per-tier confusion, and VA-scatter figures."
    fig_dir = Path(figures_dir) if figures_dir is not None else Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    accuracies = {tier: metrics[tier]["accuracy"] for tier in conditions if tier in metrics}
    figures.plot_accuracy(
        accuracies,
        fig_dir / "accuracy.png",
        chance=metrics["chance"],
        accuracy_ci={tier: metrics[tier]["accuracy_ci"] for tier in accuracies},
    )
    for tier in conditions:
        tier_records = [record for record in records if record["tier"] == tier]
        if not tier_records:
            continue
        matrix = confusion_matrix(
            _col(tier_records, "intended"), _col(tier_records, "predicted"), EMOTIONS
        )
        figures.plot_confusion(matrix, EMOTIONS, fig_dir / f"confusion_{tier}.png", title=tier)
    if any(record["tier"] == "affect" for record in records):
        figures.plot_va_scatter(records, "affect", fig_dir / "va_affect.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Affect-recovery benchmark")
    parser.add_argument("--backend", default="diffusers", choices=["diffusers", "hf-api", "null"])
    parser.add_argument("--track", default="content", choices=["content", "text"])
    parser.add_argument(
        "--benchmark", default=None, help="AffectBench CSV (required for text track)"
    )
    parser.add_argument("--contents", type=int, default=None, help="limit content-bank subjects")
    parser.add_argument("--limit", type=int, default=None, help="limit text-benchmark rows")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--style", default="artistic")
    parser.add_argument("--diffusion-model", default="stabilityai/sd-turbo")
    parser.add_argument("--probe", default="clip", choices=["clip", "hf"])
    parser.add_argument("--clip-model", default=CLIP_MODEL)
    parser.add_argument("--probe-model", default=None, help="probe model id (CLIP or HF)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; auto if unset")
    parser.add_argument("--out", default="outputs/generated/run")
    parser.add_argument(
        "--force-coverage",
        type=float,
        default=None,
        help="ablation: force the lexicon/prior blend weight (0=prior only, 1=lexicon only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="stream records to <out>/records.jsonl and resume an interrupted run",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="save measured original images under <out>/images for human rating",
    )
    args = parser.parse_args()

    result = run_experiment(
        backend=args.backend,
        track=args.track,
        benchmark=args.benchmark,
        contents=args.contents,
        limit=args.limit,
        seeds=args.seeds,
        base_seed=args.base_seed,
        style=args.style,
        out=args.out,
        width=args.width,
        height=args.height,
        device=args.device,
        diffusion_model=args.diffusion_model,
        probe=args.probe,
        clip_model=args.clip_model,
        probe_model=args.probe_model,
        coverage_override=args.force_coverage,
        resume=args.resume,
        save_images=args.save_images,
    )
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
