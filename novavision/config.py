"""Application settings and pinned model revisions."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
DIFFUSION_MODEL = "stabilityai/sd-turbo"
DIFFUSION_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


DEFAULT_HF_TIMEOUT = 120.0


def default_revision(model_id: str, default_model: str, revision: str) -> str | None:
    "Return the pinned revision only for the default model."
    return revision if model_id == default_model else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOVA_", extra="ignore")

    backend: str = "null"
    emotion_model: str = EMOTION_MODEL
    diffusion_model: str = DIFFUSION_MODEL
    hf_timeout: float = Field(default=DEFAULT_HF_TIMEOUT, gt=0, allow_inf_nan=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
