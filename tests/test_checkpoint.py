import json

import pytest
from PIL import Image

from novavision.eval.probes import Recovery, image_digest
from novavision.experiments import run
from novavision.generation.base import NullBackend

MANIFEST = {"config": {"base_seed": 0}}
RECORD = {"tier": "raw", "intended": "joy", "index": 0, "seed": 0, "predicted": "joy"}
NEXT_RECORD = {**RECORD, "tier": "naive"}


def _line(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n"


@pytest.mark.parametrize(
    "tail",
    [
        b'{"tier": "naive", "intended": "joy", "index": 0, "seed": 0, "predicted":',
        b'{"tier": "naive", "content": "caf\xc3',
    ],
)
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_resume_discards_only_partial_final_record_and_can_append(tmp_path, tail, newline):
    path = tmp_path / "records.jsonl"
    record = {**RECORD, "content": "café"}
    complete = (_line({"manifest": MANIFEST}) + _line(record)).replace(b"\n", newline)
    path.write_bytes(complete + tail)

    checkpoint = run._Checkpoint(path, MANIFEST)

    assert checkpoint.cached("raw", "joy", 0, 0) == record
    assert checkpoint.cached("naive", "joy", 0, 0) is None
    assert path.read_bytes() == complete
    checkpoint.record(NEXT_RECORD)
    reopened = run._Checkpoint(path, MANIFEST)
    assert reopened.cached("raw", "joy", 0, 0) == record
    assert reopened.cached("naive", "joy", 0, 0) == NEXT_RECORD
    assert path.read_bytes().splitlines() == (complete + _line(NEXT_RECORD)).splitlines()


@pytest.mark.parametrize("final_record", [None, {**RECORD, "content": "café"}])
def test_resume_keeps_valid_final_json_without_newline(tmp_path, final_record):
    path = tmp_path / "records.jsonl"
    contents = _line({"manifest": MANIFEST})
    if final_record is not None:
        contents += _line(final_record)
    path.write_bytes(contents[:-1])

    checkpoint = run._Checkpoint(path, MANIFEST)

    assert path.read_bytes() == contents
    if final_record is not None:
        assert checkpoint.cached("raw", "joy", 0, 0) == final_record
    checkpoint.record(NEXT_RECORD)
    reopened = run._Checkpoint(path, MANIFEST)
    assert reopened.cached("naive", "joy", 0, 0) == NEXT_RECORD
    if final_record is not None:
        assert reopened.cached("raw", "joy", 0, 0) == final_record
    assert path.read_bytes().splitlines() == (contents + _line(NEXT_RECORD)).splitlines()


@pytest.mark.parametrize("bad_line", [b'{"tier":', b'{"content": "caf\xc3'])
@pytest.mark.parametrize("following", [b"", _line(NEXT_RECORD), b'{"tier":'])
def test_resume_rejects_terminated_or_earlier_corruption(tmp_path, bad_line, following):
    path = tmp_path / "records.jsonl"
    contents = _line({"manifest": MANIFEST}) + _line(RECORD) + bad_line + b"\n" + following
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        run._Checkpoint(path, MANIFEST)

    assert path.read_bytes() == contents


def test_resume_checks_provenance_before_repairing_partial_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    contents = _line({"manifest": MANIFEST}) + _line(RECORD) + b'{"tier":'
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="provenance"):
        run._Checkpoint(path, {"config": {"base_seed": 1}})

    assert path.read_bytes() == contents


@pytest.mark.parametrize("bad_record", [{"tier": "naive"}, None])
def test_resume_rejects_valid_json_with_invalid_record_shape(tmp_path, bad_record):
    path = tmp_path / "records.jsonl"
    contents = _line({"manifest": MANIFEST}) + _line(RECORD) + _line(bad_record)[:-1]
    path.write_bytes(contents)

    with pytest.raises((KeyError, TypeError, ValueError)):
        run._Checkpoint(path, MANIFEST)

    assert path.read_bytes() == contents


def test_resume_checks_conflicts_before_repairing_partial_tail(tmp_path):
    path = tmp_path / "records.jsonl"
    contents = (
        _line({"manifest": MANIFEST})
        + _line(RECORD)
        + _line({**RECORD, "predicted": "sadness"})
        + b'{"tier":'
    )
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="conflicting"):
        run._Checkpoint(path, MANIFEST)

    assert path.read_bytes() == contents


@pytest.mark.parametrize("problem", ["missing", "changed"])
def test_resume_verifies_originals_before_repairing_partial_tail(tmp_path, problem):
    images = tmp_path / "images"
    images.mkdir()
    image = Image.new("RGB", (2, 2), "blue")
    record = {
        **RECORD,
        "image_path": "images/original.png",
        "image_pixel_sha256": image_digest(image),
    }
    if problem == "changed":
        Image.new("RGB", (2, 2), "red").save(images / "original.png")
    path = tmp_path / "records.jsonl"
    contents = _line({"manifest": MANIFEST}) + _line(record) + b'{"tier":'
    path.write_bytes(contents)

    with pytest.raises(ValueError, match="Saved original image"):
        run._Checkpoint(path, MANIFEST, images_dir=images)

    assert path.read_bytes() == contents


def test_resumed_experiment_regenerates_partial_record_and_skips_complete_record(
    tmp_path, monkeypatch
):
    class CountingBackend(NullBackend):
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return super().generate(*args, **kwargs)

    class FakeProbe:
        name = "test"

        def recover(self, image):
            return Recovery("joy", 0.5, 0.6, {"joy": 1.0})

        def clip_t(self, image, text):
            return 0.25

    monkeypatch.setattr(run, "EMOTIONS", ("joy",))
    path = tmp_path / "records.jsonl"
    path.write_bytes(_line({"manifest": MANIFEST}) + _line(RECORD) + _line(NEXT_RECORD)[:-2])
    checkpoint = run._Checkpoint(path, MANIFEST)
    backend = CountingBackend()

    records = run._content_records(
        ["a city street"], backend, FakeProbe(), "artistic", 1, 0, 8, 8, checkpoint
    )

    assert records[0] == RECORD
    assert backend.calls == len(records) - 1 == 4
    assert records[1]["tier"] == "naive"
    assert records[1]["image_pixel_sha256"]
    reopened = run._Checkpoint(path, MANIFEST)
    assert [reopened.cached(r["tier"], "joy", 0, 0) for r in records] == run.json_safe(records)
