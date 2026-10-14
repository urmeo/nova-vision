import pytest

import novavision
from novavision.affect.analyzer import EmotionAnalysis
from novavision.generation import NullBackend
from novavision.pipeline import NovaVision


class StubAnalyzer:
    def __init__(self, emotion="joy"):
        self.emotion = emotion

    def analyze(self, text):
        return EmotionAnalysis(self.emotion, 0.9, 0.8, 0.7, 1.0, {self.emotion: 0.9})


def test_run_produces_result():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer())
    result = nv.run("good news", tier="affect", seed=3)
    assert result.image.size == (512, 512)
    assert result.seed == 3
    assert result.backend == "null"
    assert "warm vibrant palette" in result.prompt


def test_auto_run_skips_neutral():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer(emotion="neutral"))
    assert nv.auto_run("the table is wooden").tier == "raw"


def test_auto_run_conditions_emotion():
    nv = NovaVision(backend=NullBackend(), analyzer=StubAnalyzer(emotion="joy"))
    assert nv.auto_run("what a wonderful day").tier == "affect"


def test_build_pipeline_returns_lazy_null_pipeline(monkeypatch):
    from novavision.config import get_settings
    from novavision.pipeline import build_pipeline

    monkeypatch.setenv("NOVA_BACKEND", "null")
    get_settings.cache_clear()
    nv = build_pipeline()
    get_settings.cache_clear()
    assert nv.backend.name == "null"
    assert nv.analyzer is not None


@pytest.mark.parametrize("backend", ["diffusers", "DIFFUSERS", "hf-api", "HF-API"])
def test_build_pipeline_honors_configured_diffusion_model(backend, monkeypatch):
    from novavision.config import Settings
    from novavision.generation import diffusers_backend
    from novavision.pipeline import build_pipeline

    monkeypatch.setenv("HF_TOKEN", "dummy")
    monkeypatch.setattr(diffusers_backend, "_pick_device", lambda: "cpu")
    settings = Settings(backend=backend, diffusion_model="organization/custom-model")

    nv = build_pipeline(settings)

    assert nv.backend.name == backend.lower()
    assert nv.backend.model_id == settings.diffusion_model
    if backend.lower() == "diffusers":
        assert nv.backend._pipe is None
        assert nv.backend.revision is None
    else:
        assert nv.backend.timeout == settings.hf_timeout


def test_public_api_is_importable():
    assert novavision.__version__ == "1.0.0"
    assert callable(novavision.build_pipeline)
    assert novavision.NovaVision is not None and novavision.Result is not None

    assert callable(novavision.run_experiment)


def test_unknown_top_level_attr_raises():
    with pytest.raises(AttributeError):
        novavision.__getattr__("not_a_real_symbol")
