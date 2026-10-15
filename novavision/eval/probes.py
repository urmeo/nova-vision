"Affect recovery probes."

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from novavision.taxonomy import (
    AROUSAL_LADDER,
    EMOTION_PROMPTS,
    EMOTIONS,
    VALENCE_LADDER,
    prior,
)

_VA_TEMPERATURE = 10.0


@dataclass(frozen=True)
class Recovery:
    emotion: str
    valence: float
    arousal: float
    scores: dict[str, float]


def image_digest(image: Image.Image) -> str:
    """Hash decoded RGB pixels and dimensions, independent of the image file format."""
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}:{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


class Probe(ABC):
    """Reads affect back from a generated image."""

    name = "base"

    @abstractmethod
    def recover(self, image: Image.Image) -> Recovery: ...

    def clip_t(self, image: Image.Image, text: str) -> float:
        return float("nan")


class HFImageClassifierProbe(Probe):
    "A non-CLIP image-emotion classifier, independent of the prompt vocabulary."

    def __init__(
        self,
        model_id: str,
        label_map: dict | None = None,
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.label_map = {k.lower(): v for k, v in (label_map or {}).items()}
        self.revision = revision
        self.device = device
        self.name = f"img:{model_id.rsplit('/', 1)[-1]}"
        self._pipe: Any = None
        self._top_k: int | None = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            pipe = pipeline(
                "image-classification",
                model=self.model_id,
                revision=self.revision,
                device=self.device if self.device is not None else -1,
            )

            self._check_label_coverage(pipe.model.config.id2label)

            self._top_k = int(pipe.model.config.num_labels)
            self._pipe = pipe

    def _check_label_coverage(self, id2label: Mapping) -> list[str]:
        """Fail at load time if no model label reaches the Ekman set."""
        mapped = sorted(
            {self.label_map.get(str(v).lower(), str(v).lower()) for v in id2label.values()}
            & set(EMOTIONS)
        )
        if not mapped:
            raise ValueError(
                f"{self.model_id}: none of its labels map to the Ekman set; pass label_map"
            )
        return mapped

    def recover(self, image: Image.Image) -> Recovery:
        self._load()
        known = set(EMOTIONS)
        scores: dict[str, float] = {}
        for pred in self._pipe(image, top_k=self._top_k):
            try:
                label, score = pred["label"], float(pred["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{self.model_id} produced an invalid image prediction") from exc
            if not isinstance(label, str) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{self.model_id} produced an invalid image label or probability")
            label = label.lower()
            label = self.label_map.get(label, label)
            if label in known:
                scores[label] = scores.get(label, 0.0) + score
        if not scores:
            raise RuntimeError(f"{self.model_id} produced no Ekman-mappable labels for this image")
        emotion = max(scores, key=scores.__getitem__)
        valence, arousal = prior(emotion)
        return Recovery(emotion, valence, arousal, scores)


def _softmax(values):
    import numpy as np

    logits = np.asarray(values, dtype=float)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


class CLIPProbe(Probe):
    "Zero-shot emotion and graded valence/arousal with CLIP."

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.name = f"clip:{model_id.rsplit('/', 1)[-1]}"
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._scale: float = 1.0
        self._emotion_features: Any = None
        self._valence_features: Any = None
        self._arousal_features: Any = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = (
                CLIPModel.from_pretrained(self.model_id, revision=self.revision)
                .to(self._device)
                .eval()
            )
            processor = CLIPProcessor.from_pretrained(self.model_id, revision=self.revision)
            scale = float(model.logit_scale.exp().item())
            self._model = model
            self._processor = processor
            self._scale = scale

    def _image_features(self, image: Image.Image):
        import torch

        self._load()
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            pooled = self._model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            features = self._model.visual_projection(pooled)
        return features / features.norm(dim=-1, keepdim=True)

    def _text_features(self, texts):
        import torch

        self._load()
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        with torch.no_grad():
            pooled = self._model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ).pooler_output
            features = self._model.text_projection(pooled)
        return features / features.norm(dim=-1, keepdim=True)

    def _fixed_features(self):
        if self._emotion_features is None:
            import torch

            means = []
            for label in EMOTIONS:
                features = self._text_features(EMOTION_PROMPTS[label])
                mean = features.mean(dim=0)
                means.append(mean / mean.norm())
            emotion_features = torch.stack(means)
            valence_features = self._text_features([prompt for prompt, _ in VALENCE_LADDER])
            arousal_features = self._text_features([prompt for prompt, _ in AROUSAL_LADDER])
            self._emotion_features = emotion_features
            self._valence_features = valence_features
            self._arousal_features = arousal_features
        return self._emotion_features, self._valence_features, self._arousal_features

    def _expected(self, image_features, text_features, ladder):
        import numpy as np

        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        probabilities = _softmax(_VA_TEMPERATURE * similarities)
        return float(np.dot(probabilities, [value for _, value in ladder]))

    def recover(self, image: Image.Image) -> Recovery:
        labels = list(EMOTIONS)
        emotion_features, valence_features, arousal_features = self._fixed_features()
        image_features = self._image_features(image)

        similarities = (image_features @ emotion_features.T).squeeze(0).cpu().numpy()
        probabilities = _softmax(self._scale * similarities)
        scores = {label: float(p) for label, p in zip(labels, probabilities)}
        emotion = labels[int(probabilities.argmax())]

        valence = self._expected(image_features, valence_features, VALENCE_LADDER)
        arousal = self._expected(image_features, arousal_features, AROUSAL_LADDER)
        return Recovery(emotion, valence, arousal, scores)

    def clip_t(self, image: Image.Image, text: str) -> float:
        image_features = self._image_features(image)
        text_features = self._text_features([text])
        return float((image_features @ text_features.T).item())
