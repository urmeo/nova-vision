"Build image prompts for each conditioning tier."

from __future__ import annotations

from novavision.taxonomy import NEUTRAL

TIERS = ("raw", "naive", "emotion", "affect")
FLOORS = ("scene",)

STYLE_PRESETS = {
    "photorealistic": "hyperrealistic photograph, 85mm lens, natural light, ultra sharp, 8k",
    "artistic": "digital painting, expressive brushwork, rich color, dramatic lighting",
    "abstract": "abstract art, geometric forms, bold composition, modern gallery piece",
    "nature": "landscape photography, golden hour, atmospheric depth, fine detail",
    "dreamscape": "surreal dreamscape, ethereal glow, volumetric light, imaginative",
}


EMOTION_MOODS = {
    "joy": "joyful uplifting mood, bright cheerful atmosphere",
    "sadness": "sad melancholic mood, somber wistful atmosphere",
    "anger": "angry intense mood, hostile aggressive atmosphere",
    "fear": "fearful anxious mood, tense ominous atmosphere",
    "surprise": "surprised startled mood, sudden astonishing atmosphere",
    "disgust": "disgusted repulsed mood, sickly off-putting atmosphere",
    "neutral": "calm neutral mood, plain everyday atmosphere",
}


EMOTION_SCENES = {
    "joy": "a radiant sunlit meadow full of blooming wildflowers",
    "sadness": "a misty rain-soaked forest at dusk",
    "anger": "a dramatic thunderstorm with lightning over a dark sky",
    "fear": "a fog-shrouded landscape under a pale moon",
    "surprise": "a sudden burst of colorful aurora across the night sky",
    "disgust": "a murky overgrown swamp with sickly green hues",
    "neutral": "a quiet zen garden at first light",
}

QUALITY = "masterpiece, highly detailed, sharp focus, professional"


VALENCE_POSITIVE, VALENCE_NEGATIVE = 0.33, -0.33
AROUSAL_HIGH, AROUSAL_LOW = 0.66, 0.33

NEGATIVE_PROMPT = (
    "watermark, text, signature, logo, blurry, low quality, distorted, "
    "deformed, bad anatomy, cropped, jpeg artifacts, lowres"
)


def va_descriptors(valence: float, arousal: float) -> str:
    "Map valence/arousal to palette and lighting cues."
    if valence >= VALENCE_POSITIVE:
        palette = "warm vibrant palette, golden tones"
    elif valence <= VALENCE_NEGATIVE:
        palette = "cool desaturated palette, muted blue-grey tones"
    else:
        palette = "balanced natural palette"

    if arousal >= AROUSAL_HIGH:
        lighting = "dramatic high-contrast lighting, dynamic composition"
    elif arousal <= AROUSAL_LOW:
        lighting = "soft gentle lighting, calm still composition"
    else:
        lighting = "even lighting, steady composition"

    return f"{palette}, {lighting}"


def build_prompt(
    content: str,
    *,
    emotion: str = NEUTRAL,
    valence: float = 0.0,
    arousal: float = 0.5,
    style: str = "artistic",
    tier: str = "affect",
) -> str:
    """Compose the generation prompt for a conditioning tier or floor."""
    if tier not in TIERS and tier not in FLOORS:
        raise ValueError(f"Unknown tier '{tier}', expected one of {TIERS + FLOORS}")

    style_description = STYLE_PRESETS.get(style.lower(), STYLE_PRESETS["artistic"])
    mood = EMOTION_MOODS.get(emotion, EMOTION_MOODS[NEUTRAL])

    if tier == "raw":
        parts = [content]
    elif tier == "naive":
        parts = [content, emotion]
    elif tier == "emotion":
        parts = [content, mood]
    elif tier == "affect":
        parts = [content, mood, va_descriptors(valence, arousal)]
    else:
        scene = EMOTION_SCENES.get(emotion, EMOTION_SCENES[NEUTRAL])
        parts = [scene]

    parts.extend((style_description, QUALITY))
    return ", ".join(part for part in parts if part)
