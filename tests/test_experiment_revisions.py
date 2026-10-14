import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from novavision.config import (
    CLIP_MODEL,
    CLIP_REVISION,
    DIFFUSION_MODEL,
    DIFFUSION_REVISION,
    EMOTION_MODEL,
    EMOTION_REVISION,
)
from novavision.experiments import run
from novavision.generation.base import NullBackend

FIXTURE = Path(__file__).parent / "fixtures" / "affectbench_sample.csv"
DIFFUSION_SHA = "a" * 40
EMOTION_SHA = "b" * 40
PROBE_SHA = "c" * 40


@pytest.fixture
def captured_run(monkeypatch):
    captured = SimpleNamespace(events=[], manifests=[])

    def backend(name, **kwargs):
        captured.events.append(("backend", {"name": name, **kwargs}))
        generator = NullBackend()
        generator.revision = kwargs.get("revision")
        generator.device = "cpu"
        generator.dtype = "float32"
        generator.steps = 2
        return generator

    def probe(**kwargs):
        captured.events.append(("probe", kwargs))
        return SimpleNamespace(name="test:probe", **kwargs)

    def analyzer(**kwargs):
        captured.events.append(("analyzer", kwargs))
        return SimpleNamespace(**kwargs)

    def write(out, records, metrics, contrasts, manifest, conditions):
        captured.manifests.append(manifest)

    monkeypatch.setattr(run, "get_backend", backend)
    monkeypatch.setattr(run, "CLIPProbe", probe)
    monkeypatch.setattr(run, "HFImageClassifierProbe", probe)
    monkeypatch.setattr(run, "EmotionAnalyzer", analyzer)
    monkeypatch.setattr(run, "set_determinism", lambda seed: None)
    monkeypatch.setattr(run, "_content_records", lambda *args: [])
    monkeypatch.setattr(run, "_text_records", lambda *args: [])
    monkeypatch.setattr(run, "_summarize", lambda *args: {})
    monkeypatch.setattr(run, "_contrasts", lambda records: {})
    monkeypatch.setattr(run, "_write", write)
    return captured


def test_custom_local_revisions_are_resolved_before_initialization(
    tmp_path, monkeypatch, captured_run
):
    revisions = {
        ("org/generator", "release"): DIFFUSION_SHA,
        ("org/emotion", "trained"): EMOTION_SHA,
        ("org/clip", "evaluation"): PROBE_SHA,
    }
    resolved = []

    def resolve(model_id, revision):
        assert not captured_run.events
        assert not (tmp_path / "run").exists()
        resolved.append((model_id, revision))
        return revisions[model_id, revision]

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    run.run_experiment(
        backend="diffusers",
        track="text",
        benchmark=str(FIXTURE),
        limit=1,
        diffusion_model="org/generator",
        diffusion_revision="release",
        emotion_model="org/emotion",
        emotion_revision="trained",
        clip_model="org/clip",
        clip_revision="evaluation",
        out=str(tmp_path / "run"),
    )

    assert set(resolved) == set(revisions)
    constructors = dict(captured_run.events)
    assert constructors["backend"]["revision"] == DIFFUSION_SHA
    assert constructors["analyzer"]["revision"] == EMOTION_SHA
    assert constructors["probe"]["revision"] == PROBE_SHA
    manifest = captured_run.manifests[0]
    assert manifest["model_revisions"] == {
        "diffusion": DIFFUSION_SHA,
        "emotion": EMOTION_SHA,
        "clip": PROBE_SHA,
        "probe": PROBE_SHA,
    }
    config = manifest["config"]
    assert config["requested_model_revisions"] == {
        "diffusion": "release",
        "emotion": "trained",
        "probe": "evaluation",
    }
    for key in ("diffusion", "emotion", "clip", "probe"):
        assert config[f"{key}_revision"] == manifest["model_revisions"][key]


def test_hf_probe_revision_is_recorded_without_claiming_a_clip_model(
    tmp_path, monkeypatch, captured_run
):
    resolved = []

    def resolve(model_id, revision):
        resolved.append((model_id, revision))
        return PROBE_SHA

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    run.run_experiment(
        backend="null",
        probe="hf",
        probe_model="org/image-emotion",
        probe_revision="validated",
        out=str(tmp_path),
    )

    assert resolved == [("org/image-emotion", "validated")]
    assert dict(captured_run.events)["probe"]["revision"] == PROBE_SHA
    manifest = captured_run.manifests[0]
    assert manifest["model_revisions"] == {
        "diffusion": None,
        "emotion": None,
        "clip": None,
        "probe": PROBE_SHA,
    }
    assert manifest["config"]["clip_revision"] is None
    assert manifest["config"]["probe_revision"] == PROBE_SHA


@pytest.mark.parametrize(
    "revision_option, requested_revision, model_id",
    [
        ("clip_revision", "experiment", "org/other-clip"),
        ("probe_revision", "experiment", "org/other-clip"),
        ("probe_revision", None, "org/other-clip"),
        ("clip_revision", "main", CLIP_MODEL),
    ],
)
def test_clip_revision_aliases_apply_to_the_selected_probe_model(
    tmp_path, monkeypatch, captured_run, revision_option, requested_revision, model_id
):
    resolved = []

    def resolve(model_id, revision):
        resolved.append((model_id, revision))
        return PROBE_SHA

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    run.run_experiment(
        backend="null",
        probe_model=model_id,
        out=str(tmp_path),
        **{revision_option: requested_revision},
    )

    assert resolved == [(model_id, requested_revision)]
    assert dict(captured_run.events)["probe"]["model_id"] == model_id
    assert captured_run.manifests[0]["model_revisions"]["clip"] == PROBE_SHA


@pytest.mark.parametrize("backend", ["null", "hf-api"])
def test_unused_models_are_not_resolved_or_recorded(tmp_path, monkeypatch, captured_run, backend):
    resolved = []

    def resolve(model_id, revision):
        resolved.append((model_id, revision))
        assert (model_id, revision) == (CLIP_MODEL, CLIP_REVISION)
        return revision

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    run.run_experiment(
        backend=backend,
        diffusion_model="org/unused-generator",
        emotion_model="org/unused-classifier",
        out=str(tmp_path),
    )

    assert resolved == [(CLIP_MODEL, CLIP_REVISION)]
    revisions = captured_run.manifests[0]["model_revisions"]
    assert revisions["diffusion"] is None
    assert revisions["emotion"] is None
    assert revisions["clip"] == revisions["probe"] == CLIP_REVISION


def test_known_model_defaults_use_their_pinned_commits(tmp_path, monkeypatch, captured_run):
    resolved = []

    def resolve(model_id, revision):
        resolved.append((model_id, revision))
        return revision

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    run.run_experiment(
        backend="diffusers",
        track="text",
        benchmark=str(FIXTURE),
        limit=1,
        out=str(tmp_path),
    )

    assert set(resolved) == {
        (DIFFUSION_MODEL, DIFFUSION_REVISION),
        (EMOTION_MODEL, EMOTION_REVISION),
        (CLIP_MODEL, CLIP_REVISION),
    }
    assert captured_run.manifests[0]["config"]["requested_model_revisions"] == {
        "diffusion": None,
        "emotion": None,
        "probe": None,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "hf-api", "diffusion_revision": "release"},
        {"backend": "null", "diffusion_revision": "release"},
        {"emotion_revision": "release"},
        {"clip_revision": "first", "probe_revision": "second"},
        {"probe": "hf", "probe_model": "org/probe", "clip_revision": "release"},
    ],
)
def test_invalid_revision_options_fail_before_resolution_or_initialization(
    tmp_path, monkeypatch, captured_run, options
):
    def unexpected_resolution(*args):
        pytest.fail("Invalid revision options attempted model resolution")

    monkeypatch.setattr(run, "resolve_model_revision", unexpected_resolution)
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        run.run_experiment(out=str(out), **options)

    assert not captured_run.events
    assert not out.exists()


def test_resolution_failure_leaves_no_models_or_checkpoint(tmp_path, monkeypatch, captured_run):
    def resolve(model_id, revision):
        if model_id == "org/missing-probe":
            raise ValueError("Model revision could not be resolved")
        return DIFFUSION_SHA

    monkeypatch.setattr(run, "resolve_model_revision", resolve)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="could not be resolved"):
        run.run_experiment(
            diffusion_model="org/generator",
            clip_model="org/missing-probe",
            out=str(out),
            resume=True,
        )

    assert not captured_run.events
    assert not out.exists()


def test_resume_rejects_a_changed_resolved_commit(tmp_path, monkeypatch, captured_run):
    resolved_sha = PROBE_SHA
    monkeypatch.setattr(run, "resolve_model_revision", lambda *args: resolved_sha)

    def interrupted_records(*args):
        checkpoint = args[-1]
        checkpoint.record({"tier": "raw", "intended": "joy", "index": 0, "seed": 0})
        raise RuntimeError("interrupted")

    monkeypatch.setattr(run, "_content_records", interrupted_records)
    options = dict(
        backend="null",
        clip_model="org/probe",
        clip_revision="main",
        out=str(tmp_path),
        resume=True,
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        run.run_experiment(**options)
    checkpoint = tmp_path / "records.jsonl"
    original = checkpoint.read_bytes()
    manifest = json.loads(original.splitlines()[0])["manifest"]
    assert manifest["model_revisions"]["probe"] == PROBE_SHA
    assert manifest["config"]["requested_model_revisions"]["probe"] == "main"

    resolved_sha = "d" * 40
    with pytest.raises(ValueError, match="provenance"):
        run.run_experiment(**options)

    assert checkpoint.read_bytes() == original
    assert not captured_run.manifests


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_hosted_deadline_fails_before_model_resolution(
    tmp_path, monkeypatch, captured_run, timeout
):
    def unexpected_resolution(*args):
        pytest.fail("Invalid deadline attempted model resolution")

    monkeypatch.setattr(run, "resolve_model_revision", unexpected_resolution)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="timeout"):
        run.run_experiment(backend="hf-api", hf_timeout=timeout, out=str(out))

    assert not captured_run.events
    assert not out.exists()


def test_cli_passes_model_revisions_and_the_emotion_model(monkeypatch):
    received = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--diffusion-revision",
            "generator-release",
            "--emotion-model",
            "org/emotion",
            "--emotion-revision",
            "classifier-release",
            "--clip-revision",
            "probe-release",
            "--probe-revision",
            "probe-release",
            "--hf-timeout",
            "7.5",
        ],
    )
    monkeypatch.setattr(run, "run_experiment", lambda **kwargs: received.append(kwargs) or {})

    run.main()

    assert received[0]["diffusion_revision"] == "generator-release"
    assert received[0]["emotion_model"] == "org/emotion"
    assert received[0]["emotion_revision"] == "classifier-release"
    assert received[0]["clip_revision"] == received[0]["probe_revision"] == "probe-release"
    assert received[0]["hf_timeout"] == 7.5
