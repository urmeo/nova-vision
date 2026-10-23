from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from novavision.affect.analyzer import EMOTION_MODEL, EmotionAnalysis, EmotionAnalyzer
from novavision.generation.base import ImageBackend, _require_nonempty_string, _validate_request
from novavision.generation.diffusers_backend import DIFFUSION_MODEL
from novavision.generation.factory import get_backend
from novavision.generation.hf_api_backend import DEFAULT_HF_TIMEOUT
from novavision.prompting import FLOORS, NEGATIVE_PROMPT, TIERS, build_prompt
from novavision.taxonomy import NEUTRAL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOVA_", extra="ignore")

    backend: str = "null"
    emotion_model: str = EMOTION_MODEL
    diffusion_model: str = DIFFUSION_MODEL
    hf_timeout: float = Field(default=DEFAULT_HF_TIMEOUT, gt=0, allow_inf_nan=False)

    @field_validator("hf_timeout", mode="before")
    @classmethod
    def _reject_boolean_timeout(cls, value: object) -> object:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("hf_timeout must be a positive finite number")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass
class Result:
    image: Image.Image
    prompt: str
    analysis: EmotionAnalysis
    tier: str
    seed: int
    backend: str


class NovaVision:

    def __init__(
        self, backend: ImageBackend | None = None, analyzer: EmotionAnalyzer | None = None
    ) -> None:
        self.backend = get_backend("null") if backend is None else backend
        self.analyzer = EmotionAnalyzer() if analyzer is None else analyzer

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
        if not isinstance(tier, str) or tier not in TIERS + FLOORS:
            raise ValueError(f"Unknown tier '{tier}', expected one of {TIERS + FLOORS}")
        return self._generate(text, style, tier, seed, width, height)

    def auto_run(
        self,
        text: str,
        *,
        style: str = "artistic",
        seed: int = 0,
        width: int = 512,
        height: int = 512,
    ) -> Result:
        return self._generate(text, style, None, seed, width, height)

    def _generate(
        self,
        text: str,
        style: str,
        tier: str | None,
        seed: int,
        width: int,
        height: int,
    ) -> Result:
        _require_nonempty_string(text, "text")
        if not isinstance(style, str):
            raise ValueError("style must be a string")
        width, height, seed = _validate_request(text, width, height, seed, NEGATIVE_PROMPT)
        analysis = self.analyzer.analyze(text)
        if tier is None:
            tier = "raw" if analysis.primary == NEUTRAL else "affect"
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
    settings = get_settings() if settings is None else settings
    analyzer = EmotionAnalyzer(settings.emotion_model)
    backend_name = settings.backend.lower()
    kwargs: dict = (
        {"model_id": settings.diffusion_model} if backend_name in {"diffusers", "hf-api"} else {}
    )
    if backend_name == "hf-api":
        kwargs["timeout"] = settings.hf_timeout
    backend = get_backend(backend_name, **kwargs)
    return NovaVision(backend=backend, analyzer=analyzer)
