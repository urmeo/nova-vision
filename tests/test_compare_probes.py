import json
from pathlib import Path

import compare_probes
import pytest


def test_mcnemar_no_discordants_is_one():
    gold = ["joy", "anger"]
    r = compare_probes.mcnemar_exact(gold, ["joy", "anger"], ["joy", "anger"])
    assert r == {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}


def test_mcnemar_one_sided_discordants():
    gold = ["joy"] * 5
    r = compare_probes.mcnemar_exact(gold, ["anger"] * 5, ["joy"] * 5)
    assert (r["only_a_correct"], r["only_b_correct"]) == (0, 5)
    assert r["p_value"] == pytest.approx(0.0625)


def test_mcnemar_symmetric_is_capped_at_one():
    gold = ["joy", "joy"]
    r = compare_probes.mcnemar_exact(gold, ["joy", "anger"], ["anger", "joy"])
    assert r["p_value"] == 1.0


def test_saved_scene_mcnemar_arithmetic_assumes_the_unverified_arrays_are_paired():
    root = Path(__file__).resolve().parents[1] / "outputs" / "results"
    a = json.loads((root / "probe_validation_scene.json").read_text())
    b = json.loads((root / "probe_validation_scene_l14.json").read_text())
    assert a["gold"] == b["gold"]
    r = compare_probes.mcnemar_exact(a["gold"], a["predictions"], b["predictions"])
    assert r["p_value"] == pytest.approx(0.0375, abs=1e-4)


@pytest.mark.parametrize(
    ("gold", "pred_a", "pred_b"),
    [(["joy"], [], ["joy"]), (["joy"], ["joy"], []), ([], ["joy"], ["joy"])],
)
def test_mcnemar_rejects_unpaired_lengths(gold, pred_a, pred_b):
    with pytest.raises(ValueError, match="equal lengths"):
        compare_probes.mcnemar_exact(gold, pred_a, pred_b)


def test_comparison_cli_rejects_disconnected_sample_count(tmp_path, monkeypatch):
    report = {"gold": ["joy"], "predictions": ["joy"], "n": 2}
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr("sys.argv", ["compare_probes.py", str(path), str(path)])
    with pytest.raises(SystemExit, match="sample count disagrees"):
        compare_probes.main()


def _report(**changes):
    return {
        "gold": ["joy", "joy"],
        "predictions": ["joy", "anger"],
        "n": 2,
        "probe": "fake",
        "accuracy": 0.5,
        "image_digests": ["a" * 64, "b" * 64],
        **changes,
    }


def _run_comparison(tmp_path, monkeypatch, report_a, report_b):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, report in zip(paths, (report_a, report_b)):
        path.write_text(json.dumps(report))
    monkeypatch.setattr("sys.argv", ["compare_probes.py", *(str(path) for path in paths)])
    compare_probes.main()


@pytest.mark.parametrize("digests", [None, [], ["a" * 64], ["a" * 64, "invalid"]])
def test_comparison_cli_requires_per_image_fingerprints(tmp_path, monkeypatch, digests):
    with pytest.raises(SystemExit, match="image fingerprints"):
        _run_comparison(tmp_path, monkeypatch, _report(), _report(image_digests=digests))


def test_comparison_cli_rejects_different_images_with_identical_gold_labels(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="different images or image order"):
        _run_comparison(
            tmp_path, monkeypatch, _report(), _report(image_digests=["b" * 64, "a" * 64])
        )


def test_comparison_cli_accepts_verified_ordered_images(tmp_path, monkeypatch, capsys):
    _run_comparison(tmp_path, monkeypatch, _report(), _report(predictions=["joy", "joy"]))
    result = json.loads(capsys.readouterr().out)
    assert result["n"] == 2
    assert result["only_a_correct"] == 0
    assert result["only_b_correct"] == 1
    assert result["b"]["accuracy"] == 1.0
