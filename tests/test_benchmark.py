from collections import Counter
from pathlib import Path

import pytest

from novavision.data import load_benchmark, load_content_bank
from novavision.taxonomy import EMOTIONS

FIXTURE = Path(__file__).parent / "fixtures" / "affectbench_sample.csv"


def test_sample_is_balanced():
    counts = Counter(row["emotion"] for row in load_benchmark(FIXTURE))
    assert set(counts) == set(EMOTIONS)
    assert set(counts.values()) == {8}


def test_rows_have_text():
    for row in load_benchmark(FIXTURE):
        assert row["text"].strip()
        assert row["emotion"] in EMOTIONS


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_benchmark("does/not/exist.csv")


def test_rejects_unknown_emotion(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("text,emotion\nhello there,excitement\n")
    with pytest.raises(ValueError):
        load_benchmark(bad)


def test_content_bank_is_neutral_and_nonempty():
    bank = load_content_bank()
    assert len(bank) >= 10
    assert all(item.strip() for item in bank)


def test_load_benchmark_tolerates_utf8_bom(tmp_path):
    benchmark = tmp_path / "bom.csv"
    benchmark.write_text("text,emotion\nhello there,joy\n", encoding="utf-8-sig")
    assert load_benchmark(benchmark) == [{"text": "hello there", "emotion": "joy"}]


def test_load_benchmark_preserves_newlines_inside_quoted_text(tmp_path):
    benchmark = tmp_path / "multiline.csv"
    benchmark.write_bytes(b'text,emotion\r\n"first line\r\nsecond line",joy\r\n')
    assert load_benchmark(benchmark) == [{"text": "first line\r\nsecond line", "emotion": "joy"}]


def test_load_benchmark_reports_physical_line_after_multiline_text(tmp_path):
    benchmark = tmp_path / "multiline.csv"
    benchmark.write_text('text,emotion\n"first line\nsecond line",joy\nbad row,unknown\n')
    with pytest.raises(ValueError, match=r"multiline\.csv:4 unknown emotion"):
        load_benchmark(benchmark)


def test_content_bank_rejects_duplicates(tmp_path):
    bank = tmp_path / "bank.txt"
    bank.write_text("a tree\na house\nA  tree\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_content_bank(bank)
