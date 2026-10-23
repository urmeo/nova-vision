from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from novavision.eval.metrics import probability
from novavision.taxonomy import EMOTIONS, prior


@dataclass(frozen=True)
class Recovery:
    emotion: str
    valence: float
    arousal: float
    scores: dict[str, float]


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    header = f"RGB:{rgb.width}:{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


class Probe(ABC):
    name = "base"

    @abstractmethod
    def recover(self, image: Image.Image) -> Recovery: ...


class HFImageClassifierProbe(Probe):
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
        predictions = self._pipe(image, top_k=self._top_k)
        if not isinstance(predictions, (list, tuple)):
            raise ValueError(f"{self.model_id} produced invalid image predictions")
        seen_labels = set()
        for pred in predictions:
            if not isinstance(pred, Mapping) or not {"label", "score"} <= pred.keys():
                raise ValueError(f"{self.model_id} produced an invalid image prediction")
            label, score = pred["label"], pred["score"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{self.model_id} produced an invalid image label or probability")
            score = probability(score, f"{self.model_id}: invalid image probability")
            label = label.lower()
            if label in seen_labels:
                raise ValueError(f"{self.model_id} produced a duplicate image label {label!r}")
            seen_labels.add(label)
            label = self.label_map.get(label, label)
            if label in known:
                scores[label] = scores.get(label, 0.0) + score
        if not scores:
            raise RuntimeError(f"{self.model_id} produced no Ekman-mappable labels for this image")
        emotion = max(scores, key=scores.__getitem__)
        valence, arousal = prior(emotion)
        return Recovery(emotion, valence, arousal, scores)
