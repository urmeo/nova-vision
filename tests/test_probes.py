import math
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from novavision.eval.probes import CLIPProbe, HFImageClassifierProbe
from novavision.taxonomy import EMOTIONS


def _img():
    return Image.new("RGB", (8, 8), (10, 20, 30))


def test_hf_probe_maps_and_aggregates():
    probe = HFImageClassifierProbe("fake/model", label_map={"happy": "joy"})
    probe._pipe = lambda img, **kw: [
        {"label": "happy", "score": 0.6},
        {"label": "joy", "score": 0.2},
        {"label": "anger", "score": 0.2},
    ]
    rec = probe.recover(_img())
    assert rec.emotion == "joy"
    assert round(rec.scores["joy"], 4) == 0.8


def test_hf_probe_drops_unknown_labels():
    probe = HFImageClassifierProbe("fake/model")
    probe._pipe = lambda img, **kw: [
        {"label": "cat", "score": 0.9},
        {"label": "fear", "score": 0.1},
    ]
    assert probe.recover(_img()).emotion == "fear"


def test_clip_t_default_is_nan():
    probe = HFImageClassifierProbe("fake/model")
    assert math.isnan(probe.clip_t(_img(), "a city street"))


def test_probe_name():
    assert HFImageClassifierProbe("org/emotion-vit").name == "img:emotion-vit"


def test_hf_probe_raises_when_nothing_maps():
    probe = HFImageClassifierProbe("fake/model")
    probe._pipe = lambda img, **kw: [{"label": "cat", "score": 0.9}]
    with pytest.raises(RuntimeError):
        probe.recover(_img())


def test_hf_probe_label_coverage_check():
    probe = HFImageClassifierProbe("fake/model", label_map={"happy": "joy"})
    assert probe._check_label_coverage({0: "happy", 1: "cat"}) == ["joy"]
    with pytest.raises(ValueError):
        HFImageClassifierProbe("fake/model")._check_label_coverage({0: "cat"})


def test_clip_load_retries_after_processor_failure(monkeypatch):
    class FakeModel:
        logit_scale = SimpleNamespace(exp=lambda: SimpleNamespace(item=lambda: 2.0))

        def to(self, device):
            return self

        def eval(self):
            return self

    attempts = 0
    processor = object()

    def load_processor(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("processor download interrupted")
        return processor

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            CLIPModel=SimpleNamespace(from_pretrained=lambda *a, **kw: FakeModel()),
            CLIPProcessor=SimpleNamespace(from_pretrained=load_processor),
        ),
    )
    probe = CLIPProbe("fake/model")
    with pytest.raises(OSError, match="interrupted"):
        probe._load()
    probe._load()
    assert attempts == 2
    assert probe._processor is processor
    assert probe._model is not None
    assert probe._scale == 2.0


def test_clip_fixed_features_retry_after_ladder_failure(monkeypatch):
    class FakeFeatures:
        def mean(self, dim):
            return self

        def norm(self):
            return 1.0

        def __truediv__(self, divisor):
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(stack=lambda means: means))
    probe = CLIPProbe("fake/model")
    attempts = 0
    features = FakeFeatures()

    def text_features(texts):
        nonlocal attempts
        attempts += 1
        if attempts == len(EMOTIONS) + 2:
            raise OSError("ladder interrupted")
        return features

    monkeypatch.setattr(probe, "_text_features", text_features)
    with pytest.raises(OSError, match="interrupted"):
        probe._fixed_features()
    emotion_features, valence_features, arousal_features = probe._fixed_features()
    assert len(emotion_features) == len(EMOTIONS)
    assert valence_features is arousal_features is features
    assert attempts == 2 * (len(EMOTIONS) + 2)
