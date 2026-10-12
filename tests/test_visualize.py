"""Check source integrity and the evidence used by the saved-results charts."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import visualize

SOURCES = Path(__file__).resolve().parents[1] / "outputs/results"


@pytest.fixture
def source_dir(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    for path in SOURCES.glob("*.json"):
        if path.name != "results.json":
            (source / path.name).write_bytes(path.read_bytes())
    return source


@pytest.fixture
def reports(source_dir):
    return visualize.load_saved_results(source_dir)


@pytest.fixture
def pilot():
    records = [
        {"tier": tier, "intended": "anger", "predicted": prediction}
        for tier in ("raw", "emotion", "affect")
        for prediction in ("anger", "joy")
    ]
    metrics = {
        tier: {
            "n": 2,
            "accuracy": 0.5,
            "accuracy_ci": [0, 1],
            "macro_f1": 0.5,
            "shuffled_control": {"p_value": 0.5},
        }
        for tier in ("raw", "emotion", "affect")
    }
    metrics["chance"] = round(1 / 7, 4)
    return {"records": records, "metrics": metrics}


@pytest.mark.parametrize("field", ["accuracy", "n", "accuracy_ci"])
def test_disconnected_pilot_data_is_rejected(reports, pilot, field):
    report = deepcopy(pilot)
    if field == "accuracy":
        report["metrics"]["emotion"][field] = 0.99
    elif field == "n":
        report["metrics"]["emotion"][field] += 1
    else:
        report["metrics"]["emotion"][field] = [0.6, 0.9]
    labels = reports["probe_validation.json"]["labels"]
    with pytest.raises(ValueError, match="disagree|confidence bounds"):
        visualize.checked_pilot(report, labels)


def test_curve_uses_only_saved_simulation_points(reports):
    report = reports["power_analysis.json"]
    figure = visualize.planning_chart(report)
    try:
        line = figure.axes[0].lines[0]
        assert list(line.get_xdata()) == [row["recovery_accuracy"] for row in report["rows"]]
        assert list(line.get_ydata()) == [row["n_for_80pct_power"] for row in report["rows"]]
        assert "simulation" in figure.texts[1].get_text()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), None])
def test_invalid_planning_probabilities_are_rejected(reports, value):
    report = deepcopy(reports["power_analysis.json"])
    report["rows"][0]["recovery_accuracy"] = value
    with pytest.raises(ValueError, match="Planning recovery_accuracy"):
        visualize.checked_planning(report)


def test_missing_or_invalid_json_has_a_useful_error(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Missing saved result.*missing.json"):
        visualize.read_report(path)
    path.write_text("{broken")
    with pytest.raises(ValueError, match="Invalid JSON.*missing.json"):
        visualize.read_report(path)


def test_exports_are_traceable_and_leave_sources_unchanged(tmp_path, source_dir):
    original = {path.name: path.read_bytes() for path in source_dir.glob("*.json")}
    destination, table = tmp_path / "charts", tmp_path / "tables.md"
    paths = visualize.render(source_dir, destination, table)
    assert len([path for path in paths if path.suffix == ".svg"]) == 2
    assert len([path for path in paths if path.suffix == ".png"]) == 2
    assert all(path.stat().st_size > 0 for path in paths)
    provenance = json.loads((destination / "sources.json").read_text())["sources_sha256"]
    assert len(provenance) == 6
    for name, digest in provenance.items():
        assert digest == hashlib.sha256(original[name]).hexdigest()
        assert (source_dir / name).read_bytes() == original[name]
    content = table.read_text()
    assert "40.25%" in content
    assert "16.51%" in content
    assert "Full seven-class corrected recovery is unavailable" in content
    assert "permutation null" in content
    assert "Pilot records (`results.json`) are unavailable" in content
    assert "## Pilot recovery" not in content


def test_missing_pilot_leaves_existing_pilot_charts_unchanged(tmp_path, source_dir):
    destination = tmp_path / "charts"
    destination.mkdir()
    originals = {
        destination / f"{chart}.{extension}": b"archived pilot chart"
        for chart in ("recovery", "confusion")
        for extension in ("svg", "png")
    }
    for path, content in originals.items():
        path.write_bytes(content)
    paths = visualize.render(source_dir, destination, tmp_path / "tables.md")
    for path, content in originals.items():
        assert path.read_bytes() == content
        assert path not in paths


def test_cli_default_table_preserves_archived_results(tmp_path, source_dir, monkeypatch):
    archived_table = tmp_path / "docs/results-tables.md"
    archived_table.parent.mkdir()
    archived_table.write_bytes(b"archived pilot table")
    monkeypatch.setattr(visualize, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["visualize.py", "--source-dir", str(source_dir)])

    visualize.main()

    assert archived_table.read_bytes() == b"archived pilot table"
    generated_table = tmp_path / "outputs/generated/results-tables.md"
    content = generated_table.read_text()
    assert "40.25%" in content
    assert "16.51%" in content
    assert "Pilot records (`results.json`) are unavailable" in content


def test_explicit_pilot_source_regenerates_pilot_charts(tmp_path, source_dir, pilot):
    (source_dir / "results.json").write_text(json.dumps(pilot))
    destination, table = tmp_path / "charts", tmp_path / "tables.md"
    paths = visualize.render(source_dir, destination, table)
    assert len([path for path in paths if path.suffix == ".svg"]) == 4
    assert len([path for path in paths if path.suffix == ".png"]) == 4
    assert "## Pilot recovery" in table.read_text()
    provenance = json.loads((destination / "sources.json").read_text())["sources_sha256"]
    assert "results.json" in provenance


def test_invalid_optional_pilot_is_rejected(source_dir):
    (source_dir / "results.json").write_text("{broken")
    with pytest.raises(ValueError, match="Invalid JSON.*results.json"):
        visualize.load_saved_results(source_dir)
