from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from novavision.affect.analyzer import EMOTION_MODEL, EMOTION_REVISION, EmotionAnalyzer
from novavision.data.loaders import (
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
    paired_binary_test,
    pearson,
    permutation_test,
    prediction_collapse,
    spearman,
)
from novavision.eval.probes import HFImageClassifierProbe, Probe
from novavision.eval.probes import image_digest as _image_digest
from novavision.experiments.checkpoint import _Checkpoint
from novavision.experiments.manifest import build_manifest, package_version
from novavision.experiments.validation import _finite_setting, _integer_setting
from novavision.generation.base import ImageBackend
from novavision.generation.diffusers_backend import DIFFUSION_MODEL, DIFFUSION_REVISION
from novavision.generation.factory import get_backend
from novavision.generation.hf_api_backend import DEFAULT_HF_TIMEOUT
from novavision.outputs import atomic_output, json_safe
from novavision.prompting import NEGATIVE_PROMPT, TIERS, build_prompt
from novavision.revisions import default_revision, resolve_model_revision
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
    return base + ((item_index * 97 + emotion_index * 13 + seed_index) % 100_000)


def _check_seed_domain(n_items: int, seeds: int) -> None:
    if n_items > SEED_MAX_ITEMS or seeds > SEED_MAX_SEEDS:
        raise ValueError(
            f"seed mixing is collision-free only up to {SEED_MAX_ITEMS} items and "
            f"{SEED_MAX_SEEDS} seeds (got {n_items} items, {seeds} seeds); "
            "shard the run over --base-seed values at least 100000 apart instead"
        )


def _shuffle_emotion(gold: str, seed: int) -> str:
    others = [e for e in EMOTIONS if e != gold]
    return others[seed % len(others)]


def _source_fingerprint() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _resolve_revision(model, revision, default_model, pinned_revision):
    reference = (
        default_revision(model, default_model, pinned_revision) if revision is None else revision
    )
    return resolve_model_revision(model, reference)


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
    diffusion_model: str = DIFFUSION_MODEL,
    diffusion_revision: str | None = None,
    probe_model: str | None = None,
    probe_revision: str | None = None,
    emotion_model: str = EMOTION_MODEL,
    emotion_revision: str | None = None,
    hf_timeout: float = DEFAULT_HF_TIMEOUT,
    coverage_override: float | None = None,
    resume: bool = False,
    save_images: bool = False,
) -> dict:
    backend = backend.lower()
    if backend not in {"null", "diffusers", "hf-api"}:
        raise ValueError(f"Unknown backend: {backend}")
    if track not in CONDITIONS:
        raise ValueError(f"Unknown track: {track}")
    if not probe_model:
        raise ValueError("An image-classifier checkpoint is required; supply --probe-model")
    if diffusion_revision is not None and backend != "diffusers":
        raise ValueError("--diffusion-revision requires the local diffusers backend")
    if emotion_revision is not None and track != "text":
        raise ValueError("--emotion-revision requires the text track")
    hf_timeout = _finite_setting(hf_timeout, "hf_timeout")
    if hf_timeout <= 0:
        raise ValueError("Hosted generation timeout must be finite and positive")
    seeds = _integer_setting(seeds, "seeds", minimum=1)
    width = _integer_setting(width, "width", minimum=1)
    height = _integer_setting(height, "height", minimum=1)
    base_seed = _integer_setting(base_seed, "base_seed")
    if contents is not None:
        contents = _integer_setting(contents, "contents", minimum=0)
    if limit is not None:
        limit = _integer_setting(limit, "limit", minimum=0)
    if coverage_override is not None:
        coverage_override = _finite_setting(coverage_override, "coverage_override")
        if not 0 <= coverage_override <= 1:
            raise ValueError("coverage_override must be between 0 and 1")
    out_dir = Path(out)
    if out_dir.is_symlink():
        raise ValueError("Output directory cannot be a symbolic link")
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
    requested_revisions = {
        "diffusion": diffusion_revision,
        "emotion": emotion_revision,
        "probe": probe_revision,
    }
    diffusion_revision = (
        _resolve_revision(diffusion_model, diffusion_revision, DIFFUSION_MODEL, DIFFUSION_REVISION)
        if backend == "diffusers"
        else None
    )
    emotion_revision = (
        _resolve_revision(emotion_model, emotion_revision, EMOTION_MODEL, EMOTION_REVISION)
        if track == "text"
        else None
    )
    probe_revision = resolve_model_revision(probe_model, probe_revision)
    set_determinism(base_seed)
    kwargs: dict = {"model_id": diffusion_model}
    if backend == "diffusers":
        kwargs["revision"] = diffusion_revision
    elif backend == "hf-api":
        kwargs["timeout"] = hf_timeout
    if device:
        kwargs["device"] = device
    gen = get_backend(backend, **kwargs)
    probe_obj = HFImageClassifierProbe(
        model_id=probe_model, device=device, revision=probe_revision
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lexicon_path = Path(os.getenv("NOVAVISION_LEXICON") or LEXICON_PATH)
    manifest = build_manifest(
        backend=backend,
        track=track,
        diffusion_model=diffusion_model,
        diffusion_revision=diffusion_revision,
        probe=probe_obj.name,
        probe_model=probe_model,
        probe_revision=probe_revision,
        emotion_model=emotion_model if track == "text" else None,
        emotion_revision=emotion_revision,
        requested_model_revisions=requested_revisions,
        hf_timeout=hf_timeout if backend == "hf-api" else None,
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

    ckpt = _Checkpoint(
        out_dir / "records.jsonl" if resume else None,
        manifest,
        images_dir=out_dir / "images" if save_images else None,
    )
    if track == "text":
        analyzer = EmotionAnalyzer(
            model_name=emotion_model,
            revision=emotion_revision,
            coverage_override=coverage_override,
        )
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
    dump_results(
        {"manifest": manifest, "metrics": metrics, "contrasts": contrasts, "records": records},
        out_dir / "results.json",
    )
    _write_figures(out_dir, records, metrics, conditions)
    if ckpt.path and ckpt.path.exists():
        ckpt.path.unlink()
    return {"metrics": metrics, "contrasts": contrasts}


@dataclass
class _Experiment:
    gen: ImageBackend
    probe: Probe
    style: str
    seeds: int
    base_seed: int
    width: int
    height: int
    checkpoint: _Checkpoint
    total: int
    step: int = field(init=False, default=0)

    def records(
        self, content, emotion, valence, arousal, *, index=0, classified=None, conditions=TIERS
    ):
        for seed_index in range(self.seeds):
            seed = _seed(self.base_seed, index, EMOTIONS.index(emotion), seed_index)
            if conditions != ("scene",):
                self.step += 1
                print(f"[{self.step}/{self.total}] {emotion} :: {content[:30]}", flush=True)
            for tier in conditions:
                intended = _shuffle_emotion(emotion, seed) if tier == "shuffled" else emotion
                done = self.checkpoint.cached(tier, intended, index, seed_index)
                if done is not None:
                    yield done
                    continue
                target_valence, target_arousal = (
                    prior(intended) if tier == "shuffled" else (valence, arousal)
                )
                prompt = build_prompt(
                    content,
                    emotion=intended,
                    valence=target_valence,
                    arousal=target_arousal,
                    style=self.style,
                    tier="emotion" if tier == "shuffled" else tier,
                )
                image = self.gen.generate(
                    prompt,
                    width=self.width,
                    height=self.height,
                    seed=_seed(self.base_seed, index, EMOTIONS.index(intended), seed_index),
                    negative_prompt=NEGATIVE_PROMPT,
                )
                digest = _image_digest(image)
                recovery = self.probe.recover(image.copy())
                record = _record(
                    self.probe.name,
                    tier,
                    content,
                    intended,
                    seed_index,
                    recovery,
                    target_valence,
                    target_arousal,
                    classified=classified,
                    index=index,
                    image_pixel_sha256=digest,
                )
                yield self.checkpoint.record(record, image=image)


def _content_records(bank, gen, probe, style, seeds, base_seed, width, height, ckpt) -> list[dict]:
    experiment = _Experiment(
        gen, probe, style, seeds, base_seed, width, height, ckpt, len(bank) * len(EMOTIONS) * seeds
    )
    records = []
    for index, content in enumerate(bank):
        for emotion in EMOTIONS:
            records.extend(experiment.records(content, emotion, *prior(emotion), index=index))
    for emotion in EMOTIONS:
        records.extend(experiment.records("", emotion, *prior(emotion), conditions=("scene",)))
    return records


def _text_records(rows, gen, probe, analyzer, style, seeds, base_seed, width, height, ckpt):
    experiment = _Experiment(
        gen, probe, style, seeds, base_seed, width, height, ckpt, len(rows) * seeds
    )
    records = []
    for index, row in enumerate(rows):
        analysis = analyzer.analyze(row["text"])
        records.extend(
            experiment.records(
                row["text"],
                row["emotion"],
                analysis.valence,
                analysis.arousal,
                index=index,
                classified=analysis.primary,
                conditions=CONDITIONS["text"],
            )
        )
    return records


def _record(
    probe,
    tier,
    content,
    emotion,
    seed_index,
    recovery,
    valence,
    arousal,
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
        correct = [int(t == p) for t, p in zip(y_true, y_pred, strict=True)]
        lo, hi = bootstrap_ci(correct)
        collapse = prediction_collapse(y_pred)
        summary[tier] = {
            "accuracy": round(accuracy(y_true, y_pred), 4),
            "accuracy_ci": [round(lo, 4), round(hi, 4)],
            "macro_f1": round(macro_f1(y_true, y_pred, EMOTIONS), 4),
            "majority_baseline": round(majority_baseline(y_true), 4),
            "collapse": {**collapse, "rate": round(collapse["rate"], 4)},
            "shuffled_control": permutation_test(y_true, y_pred),
            "n": len(tier_records),
        }
        for axis in ("valence", "arousal"):
            intended = [record[f"intended_{axis}"] for record in tier_records]
            recovered = [record[f"recovered_{axis}"] for record in tier_records]
            low, high = bootstrap_corr_ci(intended, recovered)
            summary[tier].update(
                {
                    f"{axis}_r": round(pearson(intended, recovered), 4),
                    f"{axis}_rho": round(spearman(intended, recovered), 4),
                    f"{axis}_rho_ci": [round(low, 4), round(high, 4)],
                    f"{axis}_mae": round(mae(intended, recovered), 4),
                }
            )
    summary["chance"] = round(1 / len(EMOTIONS), 4)
    summary["probe_health"] = _probe_health(records)
    classification_accuracy = _classification_accuracy(records)
    if classification_accuracy is not None:
        summary["classification_accuracy"] = classification_accuracy
    return summary


def _probe_health(records) -> dict:
    predictions = [record["predicted"] for record in records if record["tier"] in TIERS]
    collapse = prediction_collapse(predictions)
    return {
        "majority_label": collapse["label"],
        "majority_rate": round(float(collapse["rate"]), 4),
        "distinct_labels": collapse["distinct"],
        "n_labels": len(EMOTIONS),
    }


def _classification_accuracy(records) -> float | None:
    seen, intended, classified = set(), [], []
    for record in records:
        item = record.get("index", record["content"])
        if record["tier"] not in TIERS or record.get("classified") is None or item in seen:
            continue
        seen.add(item)
        intended.append(record["intended"])
        classified.append(record["classified"])
    return round(accuracy(intended, classified), 4) if intended else None


def _contrasts(records) -> dict:
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
            result = paired_binary_test(higher_scores, lower_scores)
            contrasts[f"{higher}_vs_{lower}"] = {
                **{k: v if k == "p_value" else round(v, 4) for k, v in result.items()},
                "p_method": "mcnemar_exact",
                "n_pairs": len(higher_scores),
            }
    return contrasts


def dump_results(payload: dict, path: str | Path) -> None:
    with atomic_output(path) as stream:
        stream.write(json.dumps(json_safe(payload), indent=2, allow_nan=False).encode("utf-8"))


def _write_figures(out_dir, records, metrics, conditions, *, figures_dir=None) -> None:
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
            [record["intended"] for record in tier_records],
            [record["predicted"] for record in tier_records],
            EMOTIONS,
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
    parser.add_argument("--diffusion-model", default=DIFFUSION_MODEL)
    parser.add_argument("--diffusion-revision", help="local generator Hub commit, tag, or branch")
    parser.add_argument("--emotion-model", default=EMOTION_MODEL)
    parser.add_argument("--emotion-revision", help="text classifier Hub commit, tag, or branch")
    parser.add_argument("--probe-model", required=True, help="image-classifier checkpoint")
    parser.add_argument("--probe-revision", help="selected probe Hub commit, tag, or branch")
    parser.add_argument(
        "--hf-timeout",
        type=float,
        default=DEFAULT_HF_TIMEOUT,
        help="hosted generation deadline in seconds",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; auto if unset")
    parser.add_argument("--out", default="outputs/generated/run")
    parser.add_argument(
        "--force-coverage",
        dest="coverage_override",
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
    result = run_experiment(**vars(parser.parse_args()))
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
