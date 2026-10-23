from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

from novavision.affect.lexicon import AffectLexicon
from novavision.revisions import default_revision
from novavision.taxonomy import EMOTIONS, prior

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"

MAX_LEXICON_BLEND = 0.8


def _probability(value: object, name: str) -> float:
    message = f"{name} must be numeric, finite and in [0, 1]"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        probability = float(value)
    except OverflowError as error:
        raise ValueError(message) from error
    if not 0.0 <= probability <= 1.0:
        raise ValueError(message)
    return probability


def _classifier_scores(output: object) -> dict[str, float]:
    if not isinstance(output, (list, tuple)) or len(output) != 1:
        raise ValueError("Emotion classifier must return one prediction group for one text")
    predictions = output[0]
    if not isinstance(predictions, (list, tuple)):
        raise ValueError("Emotion classifier must return a list of label scores")
    if not predictions:
        raise ValueError("Emotion classifier returned no scores")

    scores = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping) or not {"label", "score"} <= prediction.keys():
            raise ValueError("Emotion classifier predictions must include label and score")
        label = prediction["label"]
        if not isinstance(label, str) or label.strip().lower() not in EMOTIONS:
            raise ValueError(
                f"Emotion classifier returned unsupported label {label!r}; "
                f"expected one of: {', '.join(EMOTIONS)}"
            )
        label = label.strip().lower()
        if label in scores:
            raise ValueError(f"Emotion classifier returned duplicate label {label!r}")
        scores[label] = _probability(prediction["score"], f"Emotion classifier score for {label!r}")
    return scores


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
    def __init__(
        self,
        model_name: str = EMOTION_MODEL,
        lexicon: AffectLexicon | None = None,
        revision: str | None = None,
        coverage_override: float | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be a nonempty string or None")
        self.model_name = model_name
        self.revision = (
            revision
            if revision is not None
            else default_revision(model_name, EMOTION_MODEL, EMOTION_REVISION)
        )
        self.coverage_override = (
            _probability(coverage_override, "coverage_override")
            if coverage_override is not None
            else None
        )
        self._lexicon = lexicon
        self._classifier: Callable[..., object] | None = None

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
    def classifier(self) -> Callable[..., object]:
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
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        scores = _classifier_scores(self.classifier(text, truncation=True))
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
