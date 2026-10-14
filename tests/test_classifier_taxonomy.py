import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.affect.lexicon import AffectScore
from novavision.pipeline import NovaVision
from novavision.taxonomy import EMOTIONS, prior


def _install_classifier(monkeypatch, labels):
    classifier = Mock(
        model=SimpleNamespace(config=SimpleNamespace(id2label=labels)),
        return_value=[[{"label": "JOY", "score": 0.99}]],
    )
    factory = Mock(return_value=classifier)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=factory))
    return classifier, factory


def test_supported_taxonomy_loads_once_and_normalizes_predictions(monkeypatch):
    labels = {index: emotion.upper() for index, emotion in enumerate(EMOTIONS)}
    classifier, factory = _install_classifier(monkeypatch, labels)
    lexicon = Mock()
    lexicon.score.return_value = AffectScore(0.0, 0.0, 0.0)
    analyzer = EmotionAnalyzer(model_name="custom/emotions", lexicon=lexicon)

    result = analyzer.analyze("A happy day")

    assert analyzer.classifier is classifier
    factory.assert_called_once()
    assert result.primary == "joy"
    assert result.confidence == 0.99
    assert result.scores == {"joy": 0.99}
    assert (result.valence, result.arousal) == prior("joy")


@pytest.mark.parametrize(
    "labels",
    [
        {0: "POSITIVE", 1: "NEGATIVE"},
        {index: emotion for index, emotion in enumerate(EMOTIONS[:-1])},
        {
            index: "JOY" if emotion == "neutral" else emotion
            for index, emotion in enumerate(EMOTIONS)
        },
        dict(enumerate((*EMOTIONS, "optimism"))),
        {index: None if index == 0 else emotion for index, emotion in enumerate(EMOTIONS)},
        {},
        None,
    ],
)
def test_incompatible_model_taxonomy_is_not_cached_or_used(monkeypatch, labels):
    classifier, factory = _install_classifier(monkeypatch, labels)
    lexicon = Mock()
    analyzer = EmotionAnalyzer(model_name="custom/sentiment", lexicon=lexicon)

    for _ in range(2):
        with pytest.raises(ValueError, match="must define exactly these labels"):
            analyzer.analyze("A happy day")
        assert analyzer._classifier is None

    assert factory.call_count == 2
    classifier.assert_not_called()
    lexicon.score.assert_not_called()


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([{"label": "POSITIVE", "score": 0.99}], "unsupported label"),
        ([{"label": None, "score": 0.99}], "unsupported label"),
        (
            [{"label": "joy", "score": 0.7}, {"label": "JOY", "score": 0.3}],
            "duplicate label",
        ),
        ([], "no scores"),
        ([{"label": "joy", "score": float("nan")}], "finite and in"),
        ([{"label": "joy", "score": float("inf")}], "finite and in"),
        ([{"label": "joy", "score": -0.1}], "finite and in"),
        ([{"label": "joy", "score": 1.01}], "finite and in"),
    ],
)
def test_invalid_predictions_fail_before_affect_scoring(predictions, message):
    lexicon = Mock()
    analyzer = EmotionAnalyzer(lexicon=lexicon)
    analyzer._classifier = Mock(return_value=[predictions])

    with pytest.raises(ValueError, match=message):
        analyzer.analyze("A happy day")

    lexicon.score.assert_not_called()


def test_predictions_cannot_bypass_declared_model_taxonomy(monkeypatch):
    classifier, _ = _install_classifier(monkeypatch, dict(enumerate(EMOTIONS)))
    classifier.return_value = [[{"label": "POSITIVE", "score": 0.99}]]
    lexicon = Mock()
    analyzer = EmotionAnalyzer(model_name="custom/emotions", lexicon=lexicon)

    with pytest.raises(ValueError, match="unsupported label 'POSITIVE'"):
        analyzer.analyze("A happy day")

    classifier.assert_called_once()
    lexicon.score.assert_not_called()


def test_auto_run_rejects_sentiment_prediction_before_conditioning_or_rendering(monkeypatch):
    from novavision.affect import analyzer as analyzer_module

    classifier, _ = _install_classifier(monkeypatch, dict(enumerate(EMOTIONS)))
    classifier.return_value = [[{"label": "POSITIVE", "score": 0.99}]]
    lexicon = Mock()
    emotion_prior = Mock()
    monkeypatch.setattr(analyzer_module, "prior", emotion_prior)
    analyzer = EmotionAnalyzer(model_name="custom/emotions", lexicon=lexicon)
    backend = Mock()
    app = NovaVision(backend=backend, analyzer=analyzer)

    with pytest.raises(ValueError, match="unsupported label 'POSITIVE'"):
        app.auto_run("A happy day")

    lexicon.score.assert_not_called()
    emotion_prior.assert_not_called()
    backend.generate.assert_not_called()
