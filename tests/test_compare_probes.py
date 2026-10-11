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


def test_saved_scene_labels_match_and_mcnemar_is_significant():
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
