import json

import numpy as np
import power_analysis as pa


def test_no_effect_is_unreachable():
    assert pa.min_n_for_power(pa.CHANCE, pa.CHANCE, _rng()) is None


def test_stronger_effect_needs_fewer_samples():
    rng = _rng()
    weak = pa.min_n_for_power(0.20, pa.CHANCE, rng)
    strong = pa.min_n_for_power(0.40, pa.CHANCE, rng)
    assert weak is not None and strong is not None
    assert strong < weak


def test_power_rises_with_n():
    rng = _rng()
    assert pa.power_at(30, 0.30, pa.CHANCE, rng) < pa.power_at(400, 0.30, pa.CHANCE, rng)


def test_analyze_reports_every_effect():
    report = pa.analyze(ceiling=0.455, planned_n=420)
    assert report["probe_ceiling"] == 0.455
    assert len(report["rows"]) == len(pa.EFFECTS)
    assert report["method"] == "binomial_mixture_simulation"
    assert report["protocol_power"] is False
    assert "permutation" in report["assumptions"]["null"]
    assert "not power of" in pa._format(report)

    mid = next(r for r in report["rows"] if r["effect_strength"] == 0.5)
    assert mid["power_at_planned_n"] >= 0.8


def test_power_cli_defaults_preserve_published_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    published = tmp_path / "outputs" / "results"
    published.mkdir(parents=True)
    (published / "probe_validation_scene_l14.json").write_text(json.dumps({"accuracy": 0.455}))
    snapshot = published / "power_analysis.json"
    snapshot.write_bytes(b"published planning report")
    monkeypatch.setattr("sys.argv", ["power_analysis.py"])
    pa.main()
    assert snapshot.read_bytes() == b"published planning report"
    output = tmp_path / "outputs" / "generated" / "power_analysis.json"
    assert json.loads(output.read_text())["probe_ceiling"] == 0.455


def _rng():
    return np.random.default_rng(0)
