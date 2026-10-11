import json

import resummarize

from novavision.taxonomy import EMOTIONS


def _run_with_records(tmp_path):
    records = []
    for tier in ("raw", "emotion", "affect", "scene"):
        for emotion in EMOTIONS:
            records.append(
                {
                    "probe": "clip:test",
                    "tier": tier,
                    "content": "" if tier == "scene" else "a city street",
                    "intended": emotion,
                    "classified": None,
                    "seed": 0,
                    "predicted": "neutral",
                    "intended_valence": 0.2,
                    "intended_arousal": 0.6,
                    "recovered_valence": 0.05,
                    "recovered_arousal": 0.5,
                    "clip_t": 0.25,
                }
            )
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"manifest": {"git_sha": "old"}, "records": records}))
    return path


def test_resummarize_adds_collapse_diagnostic_without_touching_records(tmp_path):
    path = _run_with_records(tmp_path)
    before = json.loads(path.read_text())["records"]

    payload = resummarize.resummarize(path)

    assert payload["records"] == before

    health = payload["metrics"]["probe_health"]
    assert health["majority_label"] == "neutral"
    assert health["distinct_labels"] == 1
    assert health["majority_rate"] == 1.0

    assert "reanalysis" in payload["manifest"]
    assert payload["manifest"]["git_sha"] == "old"

    assert (tmp_path / "figures" / "accuracy.png").exists()


def test_resummarize_writes_figures_to_explicit_directory(tmp_path):
    path = _run_with_records(tmp_path)
    destination = tmp_path / "published-figures"
    resummarize.resummarize(path, figures_dir=destination)
    assert (destination / "accuracy.png").exists()
    assert not (tmp_path / "figures").exists()


def test_resummarize_cli_keeps_published_records_and_figures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    published = tmp_path / "outputs" / "results"
    published.mkdir(parents=True)
    source = _run_with_records(published)
    before = source.read_bytes()
    figure = tmp_path / "outputs" / "figures" / "accuracy.png"
    figure.parent.mkdir()
    figure.write_bytes(b"published figure")
    monkeypatch.setattr("sys.argv", ["resummarize.py"])
    resummarize.main()
    destination = tmp_path / "outputs" / "generated" / "reanalysis"
    assert source.read_bytes() == before
    assert figure.read_bytes() == b"published figure"
    assert (
        json.loads((destination / "results.json").read_text())["records"]
        == json.loads(before)["records"]
    )
    assert (destination / "figures" / "accuracy.png").is_file()
