"""Emotion classification with lexicon-based valence and arousal."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from novavision.affect.lexicon import AffectLexicon
from novavision.config import EMOTION_MODEL, EMOTION_REVISION, default_revision
from novavision.taxonomy import EMOTIONS, prior

DEFAULT_MODEL = EMOTION_MODEL


MAX_LEXICON_BLEND = 0.8


@dataclass(frozen=True)
class EmotionAnalysis:
    primary: str
    confidence: float
    valence: float
    arousal: float
    coverage: float
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


class EmotionAnalyzer:
    """Blend word-level affect scores with priors for the classified emotion."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lexicon: AffectLexicon | None = None,
        revision: str | None = None,
        coverage_override: float | None = None,
    ):
        self.model_name = model_name
        self.revision = revision or default_revision(model_name, DEFAULT_MODEL, EMOTION_REVISION)

        if coverage_override is not None and not 0.0 <= coverage_override <= 1.0:
            raise ValueError("coverage_override must be in [0, 1]")
        self.coverage_override = coverage_override
        self._lexicon = lexicon
        self._classifier = None

        self._lexicon_lock = threading.Lock()
        self._classifier_lock = threading.Lock()

    @property
    def lexicon(self) -> AffectLexicon:
        if self._lexicon is None:
            with self._lexicon_lock:
                if self._lexicon is None:
                    self._lexicon = AffectLexicon.load()
        return self._lexicon

    @property
    def classifier(self):
        if self._classifier is None:
            with self._classifier_lock:
                if self._classifier is None:
                    from transformers import pipeline

                    classifier = pipeline(
                        "text-classification",
                        model=self.model_name,
                        revision=self.revision,
                        top_k=None,
                        device=-1,
                    )
                    config = getattr(getattr(classifier, "model", None), "config", None)
                    labels = getattr(config, "id2label", None)
                    normalized = (
                        [label.strip().lower() for label in labels.values()]
                        if isinstance(labels, Mapping)
                        and all(isinstance(label, str) for label in labels.values())
                        else []
                    )
                    if len(normalized) != len(EMOTIONS) or set(normalized) != set(EMOTIONS):
                        raise ValueError(
                            f"Emotion classifier {self.model_name!r} must define exactly these "
                            f"labels: {', '.join(EMOTIONS)}"
                        )
                    self._classifier = classifier
        return self._classifier

    def analyze(self, text: str) -> EmotionAnalysis:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        predictions = self.classifier(text, truncation=True)[0]
        scores = {}
        for result in predictions:
            label = result["label"]
            if not isinstance(label, str) or label.strip().lower() not in EMOTIONS:
                raise ValueError(
                    f"Emotion classifier returned unsupported label {label!r}; "
                    f"expected one of: {', '.join(EMOTIONS)}"
                )
            label = label.strip().lower()
            if label in scores:
                raise ValueError(f"Emotion classifier returned duplicate label {label!r}")
            score = float(result["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Emotion classifier score for {label!r} must be finite and in [0, 1]"
                )
            scores[label] = score
        if not scores:
            raise ValueError("Emotion classifier returned no scores")
        primary = max(scores, key=scores.__getitem__)

        affect = self.lexicon.score(text)
        prior_valence, prior_arousal = prior(primary)
        lexicon_weight = (
            self.coverage_override
            if self.coverage_override is not None
            else min(affect.coverage, MAX_LEXICON_BLEND)
        )
        prior_weight = 1 - lexicon_weight
        valence = lexicon_weight * affect.valence + prior_weight * prior_valence
        arousal = lexicon_weight * affect.arousal + prior_weight * prior_arousal

        return EmotionAnalysis(
            primary=primary,
            confidence=scores[primary],
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            coverage=affect.coverage,
            scores=scores,
        )
