import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from novavision.eval.probes import HFImageClassifierProbe
from novavision.eval.validate_probe import main as validate_main
from novavision.experiments import run
from novavision.taxonomy import EMOTIONS
from scripts.correct_recovery import correct
from scripts.report import render
from scripts.resummarize import resummarize


def test_experiment_requires_an_explicit_classifier_before_loading_models(tmp_path, monkeypatch):
    load = Mock(side_effect=AssertionError("must not load a generator"))
    monkeypatch.setattr(run, "get_backend", load)
    destination = tmp_path / "run"

    with pytest.raises(ValueError, match="supply --probe-model"):
        run.run_experiment(backend="null", out=str(destination))

    load.assert_not_called()
    assert not destination.exists()


def test_explicit_classifier_experiment_can_be_resumed_and_reported(tmp_path, monkeypatch):
    calls = 0

    def classify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("interrupted inference")
        return [{"label": "joy", "score": 1.0}]

    classifier = Mock(side_effect=classify)

    def load(probe):
        probe._pipe = classifier
        probe._top_k = 1

    monkeypatch.setattr(HFImageClassifierProbe, "_load", load)
    monkeypatch.setattr(run, "set_determinism", lambda seed: None)
    destination = tmp_path / "run"
    settings = {
        "backend": "null",
        "contents": 1,
        "seeds": 1,
        "width": 8,
        "height": 8,
        "probe_model": "example/emotion-classifier",
        "probe_revision": "a" * 40,
        "out": str(destination),
        "resume": True,
    }

    with pytest.raises(RuntimeError, match="interrupted inference"):
        run.run_experiment(**settings)
    checkpoint = destination / "records.jsonl"
    assert len(checkpoint.read_text().splitlines()) == 4

    run.run_experiment(**settings)

    saved = json.loads((destination / "results.json").read_text())
    assert len(saved["records"]) == len(EMOTIONS) * 5
    assert all(record["predicted"] == "joy" for record in saved["records"])
    assert saved["manifest"]["config"]["probe_model"] == settings["probe_model"]
    assert saved["manifest"]["model_revisions"]["probe"] == "a" * 40
    assert classifier.call_count == len(saved["records"]) + 1
    assert not checkpoint.exists()
    table = render(saved)
    assert "fixed label priors" in table
    assert "Macro-F1" in table
    rerun = resummarize(destination / "results.json", out=tmp_path / "summary.json")
    assert rerun["records"] == saved["records"]


def test_validation_cli_measures_an_explicit_classifier(tmp_path, monkeypatch):
    images = tmp_path / "images" / "joy"
    images.mkdir(parents=True)
    Image.new("RGB", (4, 4), "yellow").save(images / "one.png")
    output = tmp_path / "validation.json"
    pipe = Mock(return_value=[{"label": "joy", "score": 1.0}])
    pipe.model = SimpleNamespace(config=SimpleNamespace(id2label={0: "joy"}, num_labels=1))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: pipe))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_probe",
            "--images-dir",
            str(images.parent),
            "--probe-model",
            "example/emotion-classifier",
            "--probe-revision",
            "a" * 40,
            "--out",
            str(output),
        ],
    )

    validate_main()

    report = json.loads(output.read_text())
    assert report["accuracy"] == 1.0
    assert report["model"] == "example/emotion-classifier"
    assert report["model_revision"] == "a" * 40
    assert report["gold"] == report["predictions"] == ["joy"]
    assert len(report["image_digests"]) == 1


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model", "example/other-classifier", "models differ"),
        ("model_revision", "b" * 40, "revisions differ"),
    ],
)
def test_recovery_correction_rejects_a_different_classifier(field, value, message):
    results = {
        "records": [],
        "manifest": {
            "config": {"probe_model": "example/emotion-classifier"},
            "model_revisions": {"probe": "a" * 40},
        },
    }
    validation = {
        "model": "example/emotion-classifier",
        "model_revision": "a" * 40,
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        correct(results, validation, "emotion")
