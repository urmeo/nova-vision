import json
from copy import deepcopy
from pathlib import Path

import pytest
import visualize as renderer

ROOT = Path(__file__).resolve().parents[1]


def test_all_saved_probe_metrics_match_predictions():
    for _, *filenames in renderer.SOURCES:
        for filename in filenames:
            report = json.loads((ROOT / "outputs/results" / filename).read_text())
            score, counts = renderer.checked_probe(report)
            assert round(score, 4) == report["accuracy"]
            assert counts.sum() == report["n"]


@pytest.mark.parametrize("field", ["accuracy", "n", "confusion", "predictions"])
def test_renderer_rejects_disconnected_metrics(field):
    report = json.loads((ROOT / "outputs/results/probe_validation_scene.json").read_text())
    broken = deepcopy(report)
    if field == "accuracy":
        broken[field] = 0.99
    elif field == "n":
        broken[field] += 1
    elif field == "confusion":
        broken[field][0][0] += 1
    else:
        broken[field][0] = "unknown"
    with pytest.raises(ValueError, match="disagree"):
        renderer.checked_probe(broken)
