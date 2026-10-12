from pathlib import Path

import pytest

from novavision.affect.analyzer import EmotionAnalysis
from novavision.eval.probes import Recovery
from novavision.experiments import run
from novavision.taxonomy import EMOTIONS

FIXTURE = Path(__file__).parent / "fixtures" / "affectbench_sample.csv"


@pytest.mark.parametrize(
    "options",
    [
        {"track": "typo"},
        {"track": "text"},
        {"seeds": 0},
        {"seeds": -1},
        {"contents": -1},
        {"limit": -1},
        {"width": 0},
        {"height": 0},
    ],
)
def test_invalid_run_settings_fail_before_backend_initialization(monkeypatch, options):
    def unexpected_backend(*args, **kwargs):
        raise AssertionError("Invalid settings initialized a backend")

    monkeypatch.setattr(run, "get_backend", unexpected_backend)
    with pytest.raises(ValueError):
        run.run_experiment(**options)


def _records(tier: str, correct: bool):
    rows = []
    for sk in range(2):
        for i, emotion in enumerate(EMOTIONS):
            predicted = emotion if correct else EMOTIONS[(i + 1) % len(EMOTIONS)]
            rows.append(
                {
                    "probe": "clip:test",
                    "tier": tier,
                    "content": "a city street",
                    "intended": emotion,
                    "classified": None,
                    "seed": sk,
                    "predicted": predicted,
                    "intended_valence": 0.5,
                    "intended_arousal": 0.6,
                    "recovered_valence": 0.4 if correct else -0.4,
                    "recovered_arousal": 0.5,
                    "clip_t": 0.25,
                }
            )
    return rows


def test_summarize_reports_ci_and_chance():
    records = _records("affect", correct=True) + _records("raw", correct=False)
    metrics = run._summarize(records, run.CONDITIONS["content"])
    assert metrics["affect"]["accuracy"] == 1.0
    assert metrics["raw"]["accuracy"] == 0.0
    assert len(metrics["affect"]["accuracy_ci"]) == 2
    assert metrics["chance"] == round(1 / len(EMOTIONS), 4)


def test_summarize_reports_probe_collapse_and_baseline():

    records = _records("raw", correct=False)
    metrics = run._summarize(records, run.CONDITIONS["content"])

    assert "majority_baseline" in metrics["raw"]

    assert metrics["raw"]["collapse"]["distinct"] >= 1

    health = metrics["probe_health"]
    assert health["n_labels"] == len(EMOTIONS)
    assert 0.0 <= health["majority_rate"] <= 1.0

    assert len(metrics["raw"]["valence_rho_ci"]) == 2
    assert len(metrics["raw"]["arousal_rho_ci"]) == 2


def test_contrasts_detect_lift():
    records = _records("emotion", correct=True) + _records("raw", correct=False)
    contrasts = run._contrasts(records)
    assert contrasts["emotion_vs_raw"]["mean_diff"] == 1.0
    assert contrasts["emotion_vs_raw"]["p_value"] < 0.05


def test_indexed_repeated_text_keeps_distinct_paired_observations():
    records = []
    for index, correct in enumerate((False, True)):
        for tier in ("raw", "emotion"):
            records.append(
                {
                    "tier": tier,
                    "content": "repeated text",
                    "index": index,
                    "intended": "joy",
                    "seed": 0,
                    "predicted": "joy" if tier == "emotion" and correct else "sadness",
                    "classified": "joy" if correct else "sadness",
                }
            )
    contrast = run._contrasts(records)["emotion_vs_raw"]
    assert contrast["mean_diff"] == 0.5
    assert contrast["ci_low"] == 0.0
    assert contrast["ci_high"] == 1.0
    assert run._classification_accuracy(records) == 0.5


def test_shuffle_emotion_avoids_gold():
    for e in EMOTIONS:
        assert run._shuffle_emotion(e, 3) != e


def test_classifier_accuracy_ignores_wrong_label_floor_regardless_of_record_order():
    record = {"index": 0, "content": "happy text", "classified": "joy"}
    shuffled = {**record, "tier": "shuffled", "intended": "sadness"}
    raw = {**record, "tier": "raw", "intended": "joy"}
    assert run._classification_accuracy([shuffled, raw]) == 1.0
    assert run._classification_accuracy([raw, shuffled]) == 1.0


class FakeProbe:
    name = "clip:fake"

    def __init__(self, *a, **k):
        pass

    def recover(self, image):
        return Recovery("joy", 0.4, 0.6, {e: 1.0 if e == "joy" else 0.0 for e in EMOTIONS})

    def clip_t(self, image, text):
        return 0.3


def test_run_experiment_content_track(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    result = run.run_experiment(backend="null", contents=2, seeds=1, out=str(tmp_path))

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "figures" / "accuracy.png").exists()
    for tier in ("raw", "emotion", "affect", "scene"):
        assert tier in result["metrics"]

    import json

    txt = (tmp_path / "results.json").read_text()
    json.loads(txt, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite")))


def test_json_safe_replaces_non_finite():
    out = run.json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": "x", "d": 2})
    assert out == {"a": None, "b": [1.0, None], "c": "x", "d": 2}
    __import__("json").dumps(out, allow_nan=False)


def test_contents_zero_means_zero_not_all(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    result = run.run_experiment(backend="null", contents=0, seeds=1, out=str(tmp_path))

    assert "raw" not in result["metrics"] and "scene" in result["metrics"]


def test_run_experiment_text_track(tmp_path, monkeypatch):
    class FakeAnalyzer:
        def __init__(self, *a, **k):
            pass

        def analyze(self, text):
            return EmotionAnalysis("joy", 0.9, 0.5, 0.6, 1.0, {"joy": 0.9})

    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    monkeypatch.setattr(run, "EmotionAnalyzer", FakeAnalyzer)
    result = run.run_experiment(
        backend="null", track="text", benchmark=str(FIXTURE), limit=4, seeds=1, out=str(tmp_path)
    )

    for tier in ("raw", "emotion", "affect", "shuffled"):
        assert tier in result["metrics"]
    assert "classification_accuracy" in result["metrics"]


def test_text_track_requires_benchmark(monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    with pytest.raises(ValueError):
        run.run_experiment(backend="null", track="text", seeds=1, out="/tmp/x")


def test_seed_injective_within_guarded_domain():
    seen = set()
    for ci in range(run.SEED_MAX_ITEMS):
        for ei in range(len(EMOTIONS)):
            for sk in range(run.SEED_MAX_SEEDS):
                seen.add(run._seed(0, ci, ei, sk))
    assert len(seen) == run.SEED_MAX_ITEMS * len(EMOTIONS) * run.SEED_MAX_SEEDS


def test_seed_domain_guard():
    run._check_seed_domain(run.SEED_MAX_ITEMS, run.SEED_MAX_SEEDS)
    for items, seeds in ((run.SEED_MAX_ITEMS + 1, 1), (1, run.SEED_MAX_SEEDS + 1)):
        with pytest.raises(ValueError):
            run._check_seed_domain(items, seeds)


def test_coverage_override_forces_blend_weight(monkeypatch):
    from novavision.affect.analyzer import EmotionAnalyzer
    from novavision.affect.lexicon import AffectScore

    class _Lex:
        def score(self, text):
            return AffectScore(1.0, 1.0, 0.5)

    a = EmotionAnalyzer(lexicon=_Lex(), coverage_override=0.0)
    a._classifier = lambda text, **kw: [[{"label": "joy", "score": 0.9}]]
    from novavision.taxonomy import prior

    pv, pa = prior("joy")
    res = a.analyze("anything")
    assert (res.valence, res.arousal) == (round(pv, 4), round(pa, 4))


def test_coverage_override_rejects_out_of_range():
    from novavision.affect.analyzer import EmotionAnalyzer

    with pytest.raises(ValueError, match="coverage_override"):
        EmotionAnalyzer(coverage_override=1.5)


def test_resume_reuses_checkpoint_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    out = tmp_path / "run"
    run.run_experiment(backend="null", contents=1, seeds=1, out=str(out), resume=True)

    assert not (out / "records.jsonl").exists()
    assert (out / "results.json").exists()


def test_resume_skips_completed_records(tmp_path):
    import json as _json

    from novavision.experiments.run import _Checkpoint

    path = tmp_path / "records.jsonl"
    rec = {"tier": "raw", "intended": "joy", "index": 0, "seed": 0, "predicted": "joy"}
    manifest = {"config": {"base_seed": 0}}
    path.write_text(_json.dumps({"manifest": manifest}) + "\n" + _json.dumps(rec) + "\n")
    ckpt = _Checkpoint(path, manifest)
    assert ckpt.cached("raw", "joy", 0, 0) == rec
    assert ckpt.cached("raw", "anger", 0, 0) is None


@pytest.mark.parametrize("changed", ["config", "source_sha256", "packages", "git_sha"])
def test_resume_refuses_changed_provenance(tmp_path, changed):
    manifest = {
        "config": {"base_seed": 0},
        "source_sha256": "abc",
        "packages": {},
        "git_sha": "old",
    }
    path = tmp_path / "records.jsonl"
    run._Checkpoint(path, manifest)
    altered = dict(manifest, **{changed: "different"})
    with pytest.raises(ValueError, match="provenance"):
        run._Checkpoint(path, altered)


def test_resume_refuses_legacy_records_without_manifest(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"tier": "raw", "intended": "joy", "index": 0, "seed": 0}\n')
    with pytest.raises(ValueError, match="provenance"):
        run._Checkpoint(path, {"config": {}})


def test_interrupted_run_resumes_unchanged_but_rejects_changed_seed(tmp_path, monkeypatch):
    class InterruptedProbe(FakeProbe):
        calls = 0

        def recover(self, image):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("interrupted")
            return super().recover(image)

    monkeypatch.setattr(run, "CLIPProbe", InterruptedProbe)
    kwargs = dict(
        backend="null", contents=1, seeds=1, width=8, height=8, out=str(tmp_path), resume=True
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_experiment(**kwargs)
    before = (tmp_path / "records.jsonl").read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        run.run_experiment(**kwargs, base_seed=1)
    with pytest.raises(ValueError, match="provenance"):
        run.run_experiment(**kwargs, save_images=True)
    assert (tmp_path / "records.jsonl").read_bytes() == before
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    run.run_experiment(**kwargs)
    assert not (tmp_path / "records.jsonl").exists()


def test_resume_binds_sampled_data_not_only_item_count(tmp_path, monkeypatch):
    class InterruptedProbe(FakeProbe):
        def recover(self, image):
            raise RuntimeError("interrupted")

    monkeypatch.setattr(run, "CLIPProbe", InterruptedProbe)
    monkeypatch.setattr(run, "load_content_bank", lambda: ["original subject"])
    kwargs = dict(backend="null", contents=1, seeds=1, out=str(tmp_path), resume=True)
    with pytest.raises(RuntimeError):
        run.run_experiment(**kwargs)
    monkeypatch.setattr(run, "load_content_bank", lambda: ["replacement subject"])
    with pytest.raises(ValueError, match="provenance"):
        run.run_experiment(**kwargs)


def test_save_images_provides_verified_originals_for_hosted_human_study(tmp_path, monkeypatch):
    import json

    from PIL import Image

    from novavision.eval import human_study
    from novavision.generation.base import NullBackend

    monkeypatch.setattr(run, "get_backend", lambda *a, **kw: NullBackend())
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    measured = []

    class MeasuredProbe(FakeProbe):
        def recover(self, image):
            measured.append(run._image_digest(image))
            return super().recover(image)

    monkeypatch.setattr(run, "CLIPProbe", MeasuredProbe)
    run.run_experiment(
        backend="hf-api",
        contents=1,
        seeds=1,
        width=8,
        height=8,
        out=str(tmp_path),
        save_images=True,
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["manifest"]["config"]["save_images"] is True
    assert len(measured) == len(payload["records"])
    for record, digest in zip(payload["records"], measured):
        source = tmp_path / record["image_path"]
        assert record["image_path"].startswith("images/")
        with Image.open(source) as image:
            assert run._image_digest(image) == digest == record["image_pixel_sha256"]
    monkeypatch.setattr(
        human_study,
        "get_backend",
        lambda *a, **kw: pytest.fail("originals must avoid model loading"),
    )
    study = human_study.build_sheet(tmp_path, n=2, out=tmp_path / "ratings")
    assert (study / "ratings_template.csv").exists()


@pytest.mark.parametrize("problem", [None, "missing", "changed"])
def test_resume_preserves_and_verifies_saved_originals(tmp_path, monkeypatch, problem):
    import json

    from PIL import Image

    class InterruptedProbe(FakeProbe):
        calls = 0

        def recover(self, image):
            type(self).calls += 1
            if type(self).calls == 2:
                raise RuntimeError("interrupted")
            return super().recover(image)

    monkeypatch.setattr(run, "CLIPProbe", InterruptedProbe)
    kwargs = dict(
        backend="null",
        contents=1,
        seeds=1,
        width=8,
        height=8,
        out=str(tmp_path),
        resume=True,
        save_images=True,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_experiment(**kwargs)
    lines = (tmp_path / "records.jsonl").read_text().splitlines()
    cached = json.loads(lines[1])
    saved = tmp_path / cached["image_path"]
    original = saved.read_bytes()
    if problem == "missing":
        saved.unlink()
    elif problem == "changed":
        Image.new("RGB", (8, 8), "red").save(saved)
    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    if problem:
        with pytest.raises(ValueError, match="Saved original image"):
            run.run_experiment(**kwargs)
        assert not (tmp_path / "results.json").exists()
    else:
        run.run_experiment(**kwargs)
        assert saved.read_bytes() == original
        payload = json.loads((tmp_path / "results.json").read_text())
        assert payload["records"][0] == cached


def test_default_run_keeps_images_unstored(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(run, "CLIPProbe", FakeProbe)
    run.run_experiment(backend="null", contents=0, seeds=1, width=8, height=8, out=str(tmp_path))
    records = json.loads((tmp_path / "results.json").read_text())["records"]
    assert all("image_path" not in r for r in records)
    assert not (tmp_path / "images").exists()


def test_cli_can_save_original_images(monkeypatch):
    import sys

    received = []
    monkeypatch.setattr(sys, "argv", ["run", "--save-images"])
    monkeypatch.setattr(run, "run_experiment", lambda **kw: received.append(kw) or {})
    run.main()
    assert received[0]["save_images"] is True
