import sys
from pathlib import Path

import pytest

from novavision.data import build_benchmark
from novavision.data.build_benchmark import _curate, _drop_overlap, _interleave, _normalize, build
from novavision.taxonomy import EMOTIONS


def _examples():
    rows = []
    for emotion in EMOTIONS:
        for index in range(5):
            rows.append((f"{emotion} sentence number {index}", emotion))
    return rows


def test_curate_dedups_exact():
    rows = [("Same Text", "joy"), ("same   text", "joy"), ("other", "joy")]
    sampled = _curate(rows, n_per_class=10, seed=0)
    assert len(sampled["joy"]) == 2


def test_curate_balances_per_class():
    sampled = _curate(_examples(), n_per_class=3, seed=0)
    assert all(len(sampled[emotion]) == 3 for emotion in EMOTIONS)


def test_interleave_prefix_is_stratified():
    sampled = _curate(_examples(), n_per_class=3, seed=0)
    rows = _interleave(sampled)

    assert {emotion for _, emotion in rows[: len(EMOTIONS)]} == set(EMOTIONS)


def test_drop_overlap_removes_cross_split_leakage():
    examples = [("Loved it!", "joy"), ("so scared", "fear"), ("unique line", "joy")]
    train_norms = {_normalize("loved it!"), _normalize("SO   scared")}
    kept = _drop_overlap(examples, train_norms)

    assert kept == [("unique line", "joy")]

    assert _drop_overlap(examples, set()) == examples


def test_cli_threads_revision_into_build(monkeypatch):
    captured = {}

    def fake_build(n, out, seed, split, *, revision, drop_train_overlap):
        captured["revision"] = revision
        return Path(out)

    monkeypatch.setattr(build_benchmark, "build", fake_build)
    monkeypatch.setattr(sys, "argv", ["build_benchmark", "--revision", "deadbeef"])
    build_benchmark.main()
    assert captured["revision"] == "deadbeef"


def test_build_rejects_nonpositive_n():
    for bad in (0, -5):
        with pytest.raises(ValueError, match="positive integer"):
            build(n_per_class=bad, out_path="/tmp/should_not_write.csv")
