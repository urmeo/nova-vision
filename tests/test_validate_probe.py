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
