import json
from pathlib import Path

import correct_recovery as cr
import pytest

from novavision.taxonomy import EMOTIONS


def _identity_confusion(n_per_class=10):
    return [
        [n_per_class if i == j else 0 for j in range(len(EMOTIONS))] for i in range(len(EMOTIONS))
    ]


def _records(recovery_rate):
    records = []
    for emotion in EMOTIONS:
        incorrect = "neutral" if emotion != "neutral" else "anger"
        for sample in range(10):
            prediction = emotion if sample < recovery_rate * 10 else incorrect
            records.append({"tier": "emotion", "intended": emotion, "predicted": prediction})
    return records


def test_perfect_probe_leaves_recovery_unchanged():
    results = {"records": _records(0.5)}
    validation = {"confusion": _identity_confusion(), "model": "perfect"}
    out = cr.correct(results, validation, "emotion")

    assert out["corrected_recovery"] == out["apparent_recovery"]


def test_correction_cli_defaults_preserve_published_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    published = tmp_path / "outputs" / "results"
    published.mkdir(parents=True)
    (published / "results.json").write_text(json.dumps({"records": _records(0.5)}))
    (published / "probe_validation_scene.json").write_text(
        json.dumps({"confusion": _identity_confusion()})
    )
    snapshot = published / "corrected_recovery.json"
    snapshot.write_bytes(b"published correction")
    monkeypatch.setattr("sys.argv", ["correct_recovery.py"])
    cr.main()
    assert snapshot.read_bytes() == b"published correction"
    output = tmp_path / "outputs" / "generated" / "corrected_recovery.json"
    assert json.loads(output.read_text())["corrected_recovery"] == 0.5


def test_sensitivity_specificity_from_confusion():
    ss = cr._sensitivity_specificity(_identity_confusion())
    for e in EMOTIONS:
        sens, spec = ss[e]
        assert sens == 1.0 and spec == 1.0


def test_labelled_validation_can_use_a_different_class_order():
    labels = list(EMOTIONS)
    confusion = _identity_confusion()

    confusion[0][0] = 20
    confusion[0][1] = 5
    results = {"records": _records(0.5)}
    canonical = cr.correct(results, {"confusion": confusion, "labels": labels}, "emotion")
    reversed_report = cr.correct(
        results,
        {"confusion": [row[::-1] for row in confusion[::-1]], "labels": labels[::-1]},
        "emotion",
    )
    assert reversed_report == canonical


def test_correction_rejects_mismatched_probe_models():
    results = {"records": _records(0.5), "manifest": {"config": {"clip_model": "org/run-probe"}}}
    with pytest.raises(ValueError, match="probe models differ"):
        cr.correct(
            results, {"confusion": _identity_confusion(), "model": "org/other-probe"}, "emotion"
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_correction_rejects_invalid_confusion_counts(bad):
    confusion = _identity_confusion()
    confusion[0][0] = bad
    with pytest.raises(ValueError, match="finite nonnegative 7x7"):
        cr.correct({"records": _records(0.5)}, {"confusion": confusion}, "emotion")


def test_correction_rejects_wrong_matrix_shape():
    with pytest.raises(ValueError, match="finite nonnegative 7x7"):
        cr.correct({"records": _records(0.5)}, {"confusion": [[1]]}, "emotion")


def test_zero_support_class_serializes_null_not_nan():
    n = len(EMOTIONS)
    conf = [[0] * n for _ in range(n)]
    for i, e in enumerate(EMOTIONS):
        if e != "neutral":
            conf[i][i] = 10
    out = cr.correct({"records": _records(0.5)}, {"confusion": conf, "model": "x"}, "emotion")
    assert out["per_class"]["neutral"]["sensitivity"] is None
    assert out["corrected_recovery"] is None
    assert out["coverage"]["estimable"] == 6
    assert out["estimable_class_comparison"]["scope"] == "partial"
    json.dumps(out, allow_nan=False)


def test_committed_pilot_correction_reports_partial_common_class_comparison():
    root = Path(cr.__file__).resolve().parents[1] / "outputs" / "results"
    results = json.loads((root / "results.json").read_text())
    validation = json.loads((root / "probe_validation_scene.json").read_text())
    out = cr.correct(results, validation, "emotion")

    assert out["corrected_recovery"] is None
    assert out["apparent_recovery"] == 0.2143
    assert out["coverage"] == {
        "estimable": 5,
        "total": 7,
        "unestimable_classes": ["neutral", "surprise"],
    }
    comparison = out["estimable_class_comparison"]
    assert comparison["scope"] == "partial"
    assert comparison["labels"] == ["anger", "disgust", "fear", "joy", "sadness"]
    assert comparison["apparent_recovery"] == 0.1
    assert comparison["corrected_recovery"] == 0.1651
    assert out["per_class"]["neutral"]["status"] == "no_validation_support"
    assert out["per_class"]["surprise"]["status"] == "non_discriminating_probe"
