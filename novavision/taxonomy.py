from __future__ import annotations

EMOTIONS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

NEUTRAL = "neutral"


EMOTION_PRIORS: dict[str, tuple[float, float]] = {
    "joy": (0.80, 0.70),
    "sadness": (-0.70, 0.30),
    "anger": (-0.60, 0.85),
    "fear": (-0.75, 0.80),
    "surprise": (0.35, 0.85),
    "disgust": (-0.60, 0.50),
    "neutral": (0.00, 0.30),
}


GOEMOTIONS_TO_EKMAN: dict[str, str] = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "relief": "joy",
    "pride": "joy",
    "admiration": "joy",
    "desire": "joy",
    "caring": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "neutral": "neutral",
}


def to_ekman(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    return GOEMOTIONS_TO_EKMAN.get(label.lower().strip(), NEUTRAL)


def prior(emotion: str) -> tuple[float, float]:
    if not isinstance(emotion, str):
        raise ValueError("emotion must be a string")
    return EMOTION_PRIORS.get(emotion.lower().strip(), (0.0, 0.5))
