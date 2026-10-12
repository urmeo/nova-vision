"Saved metrics must regenerate from records, including degenerate probe outputs."

import json

from novavision.eval.probes import Recovery
from novavision.experiments import run as run_mod
from novavision.taxonomy import EMOTIONS


def _canonical(obj):
    """RFC-8259-safe, order-independent form for exact comparison."""
    return json.loads(json.dumps(run_mod.json_safe(obj), sort_keys=True))


def test_saved_metrics_regenerate_from_records(tmp_path, monkeypatch):
    class NeutralProbe:
        name = "clip:test"

        def recover(self, image):
            return Recovery("neutral", 0.0, 0.5, {"neutral": 1.0})

        def clip_t(self, image, text):
            return 0.0

    monkeypatch.setattr(run_mod, "CLIPProbe", lambda **kwargs: NeutralProbe())
    monkeypatch.setattr(run_mod, "load_content_bank", lambda: ["synthetic subject"])
    run_mod.run_experiment(
        backend="null", contents=1, seeds=1, width=8, height=8, out=str(tmp_path)
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    records = payload["records"]

    metrics = run_mod._summarize(records, run_mod.CONDITIONS["content"])
    contrasts = run_mod._contrasts(records)

    assert _canonical(metrics) == payload["metrics"]
    assert _canonical(contrasts) == payload["contrasts"]


def test_collapsed_probe_does_not_report_signal():
    records = [
        {
            "tier": tier,
            "content": "synthetic subject",
            "intended": emotion,
            "predicted": "neutral",
            "seed": 0,
            "intended_valence": (index - 3) / 3,
            "intended_arousal": index / 6,
            "recovered_valence": 0.0,
            "recovered_arousal": 0.5,
            "clip_t": 0.0,
        }
        for tier in run_mod.TIERS
        for index, emotion in enumerate(EMOTIONS)
    ]
    metrics = run_mod._summarize(records, run_mod.CONDITIONS["content"])
    for tier in run_mod.TIERS:
        assert metrics[tier]["shuffled_control"]["p_value"] == 1.0
        assert metrics[tier]["accuracy"] == metrics["chance"]

    assert metrics["probe_health"]["distinct_labels"] == 1
    assert metrics["probe_health"]["majority_label"] == "neutral"
    assert metrics["probe_health"]["majority_rate"] == 1.0
