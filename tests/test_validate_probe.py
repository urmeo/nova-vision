import json

import pytest
from PIL import Image

from novavision.eval import validate_probe
from novavision.eval.probes import Recovery
from novavision.taxonomy import EMOTIONS


class FakeProbe:
    name = "clip:fake"

    def recover(self, image):
        return Recovery("joy", 0.1, 0.5, {e: 0.0 for e in EMOTIONS})


def _make_dataset(root):
    for e in ("joy", "anger"):
        d = root / e
        d.mkdir(parents=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(d / "a.png")


def test_load_image_folder(tmp_path):
    _make_dataset(tmp_path)
    pairs = validate_probe.load_image_folder(tmp_path)
    assert len(pairs) == 2
    assert {label for _, label in pairs} == {"joy", "anger"}


def test_load_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        validate_probe.load_image_folder(tmp_path)


def test_load_image_folder_skips_directories_with_image_extensions(tmp_path):
    _make_dataset(tmp_path)
    (tmp_path / "joy" / "not-an-image.png").mkdir()
    pairs = validate_probe.load_image_folder(tmp_path)
    assert len(pairs) == 2
    assert all(path.is_file() for path, _ in pairs)


def test_validate_reports_accuracy(tmp_path):
    _make_dataset(tmp_path)
    report = validate_probe.validate(FakeProbe(), validate_probe.load_image_folder(tmp_path))
    assert report["n"] == 2
    assert report["accuracy"] == 0.5
    assert len(report["confusion"]) == len(EMOTIONS)


def test_validate_accepts_pil_images():
    pairs = [(Image.new("RGB", (8, 8)), "joy"), (Image.new("RGB", (8, 8)), "fear")]
    report = validate_probe.validate(FakeProbe(), pairs)
    assert report["n"] == 2
    assert report["accuracy"] == 0.5


def test_validate_records_ordered_decoded_image_fingerprints(tmp_path):
    first = Image.new("RGB", (8, 8), "blue")
    second = Image.new("RGB", (8, 8), "red")
    path = tmp_path / "first.png"
    first.save(path)
    in_memory = validate_probe.validate(FakeProbe(), [(first, "joy"), (second, "fear")])
    from_file = validate_probe.validate(FakeProbe(), [(path, "joy"), (second, "fear")])
    assert in_memory["image_digests"] == from_file["image_digests"]
    assert len(set(in_memory["image_digests"])) == 2
    reversed_report = validate_probe.validate(FakeProbe(), [(second, "fear"), (first, "joy")])
    assert reversed_report["image_digests"] == in_memory["image_digests"][::-1]


def test_validate_rejects_empty_sample():
    with pytest.raises(ValueError, match="at least one labelled image"):
        validate_probe.validate(FakeProbe(), [])


@pytest.mark.parametrize("n", [0, -1])
def test_load_hf_dataset_rejects_invalid_sample_size_before_loading(n):
    with pytest.raises(ValueError, match="Sample size must be positive"):
        validate_probe.load_hf_dataset("unused", n=n)


def test_ekman_aliases_cover_common_labels():
    aliases = validate_probe.EKMAN_ALIASES
    assert aliases["happy"] == "joy" and aliases["happiness"] == "joy"
    assert aliases["awe"] == "surprise" and aliases["sad"] == "sadness"
    assert set(aliases.values()) <= set(EMOTIONS)


def test_pinned_clip_revision_only_applies_to_its_model():
    import argparse

    from novavision.config import CLIP_REVISION
    from novavision.eval.validate_probe import clip_revision_for

    ns = argparse.Namespace(clip_model="openai/clip-vit-base-patch32", clip_revision=None)
    assert clip_revision_for(ns) == CLIP_REVISION
    ns = argparse.Namespace(clip_model="openai/clip-vit-large-patch14", clip_revision=None)
    assert clip_revision_for(ns) is None
    ns = argparse.Namespace(clip_model="openai/clip-vit-large-patch14", clip_revision="abc")
    assert clip_revision_for(ns) == "abc"


class _StubDS:
    def __init__(self, rows, features):
        self._rows = rows
        self.features = features

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]


def _stub_datasets(monkeypatch, rows):

    import sys
    import types

    mod = types.SimpleNamespace(load_dataset=lambda *a, **k: _StubDS(rows, {}))
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_load_hf_dataset_maps_string_label_column(monkeypatch):
    rows = [
        {"image": Image.new("RGB", (4, 4)), "emotion": "amusement", "label": 0},
        {"image": Image.new("RGB", (4, 4)), "emotion": "awe", "label": 1},
    ]
    _stub_datasets(monkeypatch, rows)
    pairs = validate_probe.load_hf_dataset("stub", n=2, label_key="emotion")
    assert sorted(e for _, e in pairs) == ["joy", "surprise"]


def test_load_hf_dataset_fails_loudly_when_nothing_maps(monkeypatch):

    rows = [{"image": Image.new("RGB", (4, 4)), "label": 0}]
    _stub_datasets(monkeypatch, rows)
    with pytest.raises(ValueError, match="label-key"):
        validate_probe.load_hf_dataset("stub", n=1)


@pytest.mark.parametrize("kind", ["clip", "hf"])
def test_validation_cli_pins_and_records_resolved_probe_revision(tmp_path, monkeypatch, kind):
    from novavision.eval import probes

    calls = []
    resolved = "a" * 40
    model = "org/image-model" if kind == "hf" else "org/clip-model"

    class PinnedProbe(FakeProbe):
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(probes, "CLIPProbe", PinnedProbe)
    monkeypatch.setattr(probes, "HFImageClassifierProbe", PinnedProbe)
    monkeypatch.setattr(
        validate_probe, "load_image_folder", lambda root: [(Image.new("RGB", (8, 8)), "joy")]
    )
    resolutions = []

    def resolve(model_id, revision):
        resolutions.append((model_id, revision))
        return resolved

    monkeypatch.setattr(validate_probe, "resolve_model_revision", resolve)
    out = tmp_path / "validation.json"
    model_flag = "--probe-model" if kind == "hf" else "--clip-model"
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_probe.py",
            "--images-dir",
            "unused",
            "--probe",
            kind,
            model_flag,
            model,
            "--probe-revision",
            "release",
            "--out",
            str(out),
        ],
    )
    validate_probe.main()

    assert resolutions == [(model, "release")]
    assert calls == [{"model_id": model, "revision": resolved}]
    report = json.loads(out.read_text())
    assert report["model"] == model
    assert report["model_revision"] == resolved
    assert report["requested_model_revision"] == "release"


def test_validation_cli_default_probe_is_pinned_without_hub_access(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from novavision.config import CLIP_MODEL, CLIP_REVISION
    from novavision.eval import probes

    calls = []

    class PinnedProbe(FakeProbe):
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(model_info=lambda *a, **kw: pytest.fail("Hub contacted")),
    )
    monkeypatch.setattr(probes, "CLIPProbe", PinnedProbe)
    monkeypatch.setattr(
        validate_probe, "load_image_folder", lambda root: [(Image.new("RGB", (8, 8)), "joy")]
    )
    out = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv", ["validate_probe.py", "--images-dir", "unused", "--out", str(out)]
    )

    validate_probe.main()

    assert calls == [{"model_id": CLIP_MODEL, "revision": CLIP_REVISION}]
    report = json.loads(out.read_text())
    assert report["model_revision"] == CLIP_REVISION
    assert report["requested_model_revision"] is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["--probe", "hf"],
        ["--probe", "hf", "--probe-model", "org/model", "--clip-revision", "main"],
        ["--probe-model", "org/model"],
        ["--clip-revision", "main", "--probe-revision", "release"],
        ["--hf-dataset", "org/dataset"],
    ],
)
def test_validation_cli_rejects_ignored_or_conflicting_options_before_loading(
    monkeypatch, arguments
):
    monkeypatch.setattr(
        validate_probe, "resolve_model_revision", lambda *a: pytest.fail("model resolved")
    )
    monkeypatch.setattr("sys.argv", ["validate_probe.py", "--images-dir", "unused", *arguments])
    with pytest.raises(SystemExit) as exc:
        validate_probe.main()
    assert exc.value.code == 2
