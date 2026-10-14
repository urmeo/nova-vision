"""End-to-end text-to-image pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from novavision.affect.analyzer import EmotionAnalysis, EmotionAnalyzer
from novavision.config import Settings, get_settings
from novavision.generation import ImageBackend, get_backend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import NEUTRAL


@dataclass
class Result:
    image: Image.Image
    prompt: str
    analysis: EmotionAnalysis
    tier: str
    seed: int
    backend: str


class NovaVision:
    """Analyze text, condition a prompt, generate an image."""

    def __init__(
        self, backend: ImageBackend | None = None, analyzer: EmotionAnalyzer | None = None
    ):
        self.backend = backend or get_backend("null")
        self.analyzer = analyzer or EmotionAnalyzer()

    def run(
        self,
        text: str,
        *,
        style: str = "artistic",
        tier: str = "affect",
        seed: int = 0,
        width: int = 512,
        height: int = 512,
    ) -> Result:
        analysis = self.analyzer.analyze(text)
        return self._render(text, analysis, style, tier, seed, width, height)

    def auto_run(
        self,
        text: str,
        *,
        style: str = "artistic",
        seed: int = 0,
        width: int = 512,
        height: int = 512,
    ) -> Result:
        "Skip emotion conditioning for neutral text."
        analysis = self.analyzer.analyze(text)
        tier = "raw" if analysis.primary == NEUTRAL else "affect"
        return self._render(text, analysis, style, tier, seed, width, height)

    def _render(
        self,
        text: str,
        analysis: EmotionAnalysis,
        style: str,
        tier: str,
        seed: int,
        width: int,
        height: int,
    ) -> Result:
        prompt = build_prompt(
            text,
            emotion=analysis.primary,
            valence=analysis.valence,
            arousal=analysis.arousal,
            style=style,
            tier=tier,
        )
        image = self.backend.generate(
            prompt, width=width, height=height, seed=seed, negative_prompt=NEGATIVE_PROMPT
        )
        return Result(image, prompt, analysis, tier, seed, self.backend.name)


def build_pipeline(settings: Settings | None = None) -> NovaVision:
    "Construct the app and CLI pipeline from shared settings."
    settings = settings or get_settings()
    backend_name = settings.backend.lower()
    kwargs: dict = (
        {"model_id": settings.diffusion_model} if backend_name in {"diffusers", "hf-api"} else {}
    )
    if backend_name == "hf-api":
        kwargs["timeout"] = settings.hf_timeout
    backend = get_backend(backend_name, **kwargs)
    return NovaVision(backend=backend, analyzer=EmotionAnalyzer(settings.emotion_model))
