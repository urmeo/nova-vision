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
def reports():
    return visualize.load_saved_results(SOURCES)


@pytest.mark.parametrize("field", ["accuracy", "n", "accuracy_ci"])
def test_disconnected_pilot_data_is_rejected(reports, field):
    report = deepcopy(reports["results.json"])
    if field == "accuracy":
        report["metrics"]["emotion"][field] = 0.99
    elif field == "n":
        report["metrics"]["emotion"][field] += 1
    else:
        report["metrics"]["emotion"][field] = [0.5, 0.9]
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


def test_exports_are_traceable_and_leave_sources_unchanged(tmp_path):
    original = {path.name: path.read_bytes() for path in SOURCES.glob("*.json")}
    destination, table = tmp_path / "charts", tmp_path / "tables.md"
    paths = visualize.render(SOURCES, destination, table)
    assert len([path for path in paths if path.suffix == ".svg"]) == 4
    assert len([path for path in paths if path.suffix == ".png"]) == 4
    assert all(path.stat().st_size > 0 for path in paths)
    provenance = json.loads((destination / "sources.json").read_text())["sources_sha256"]
    assert len(provenance) == 7
    for name, digest in provenance.items():
        assert digest == hashlib.sha256(original[name]).hexdigest()
        assert (SOURCES / name).read_bytes() == original[name]
    content = table.read_text()
    assert "40.25%" in content
    assert "16.51%" in content
    assert "Full seven-class corrected recovery is unavailable" in content
    assert "permutation null" in content
