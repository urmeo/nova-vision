import math

import pytest

from novavision.affect.analyzer import EmotionAnalyzer
from novavision.affect.lexicon import AffectLexicon, AffectScore
from novavision.taxonomy import prior


class StubLexicon(AffectLexicon):
    """A tiny lexicon with controllable coverage for testing the blend."""

    def __init__(self, valence, arousal, coverage):
        self.affect = AffectScore(valence, arousal, coverage)

    def score(self, text):
        return self.affect


def _analyzer(scores, lexicon):
    analyzer = EmotionAnalyzer(lexicon=lexicon)
    analyzer._classifier = lambda text, **kwargs: [
        [{"label": label, "score": score} for label, score in scores.items()]
    ]
    return analyzer


def test_blend_is_coverage_weighted():
    lexicon = StubLexicon(valence=1.0, arousal=1.0, coverage=0.5)
    analyzer = _analyzer({"joy": 0.9, "anger": 0.1}, lexicon)
    result = analyzer.analyze("anything")
    prior_valence, prior_arousal = prior("joy")
    assert result.primary == "joy"
    assert math.isclose(result.valence, 0.5 * 1.0 + 0.5 * prior_valence, abs_tol=1e-4)
    assert math.isclose(result.arousal, 0.5 * 1.0 + 0.5 * prior_arousal, abs_tol=1e-4)
    assert result.coverage == 0.5


def test_zero_coverage_falls_back_to_prior():
    lexicon = StubLexicon(valence=0.0, arousal=0.0, coverage=0.0)
    analyzer = _analyzer({"sadness": 0.8, "joy": 0.2}, lexicon)
    result = analyzer.analyze("zzzz oovword")
    prior_valence, prior_arousal = prior("sadness")
    assert (result.valence, result.arousal) == (round(prior_valence, 4), round(prior_arousal, 4))


def test_full_coverage_is_capped():
    lexicon = StubLexicon(valence=-0.4, arousal=0.9, coverage=1.0)
    analyzer = _analyzer({"fear": 0.7, "joy": 0.3}, lexicon)
    result = analyzer.analyze("all words known")
    prior_valence, prior_arousal = prior("fear")
    assert math.isclose(result.valence, 0.8 * -0.4 + 0.2 * prior_valence, abs_tol=1e-4)
    assert math.isclose(result.arousal, 0.8 * 0.9 + 0.2 * prior_arousal, abs_tol=1e-4)
    assert result.coverage == 1.0


def test_scores_are_read_only():
    analyzer = _analyzer({"joy": 1.0}, StubLexicon(0, 0, 0))
    result = analyzer.analyze("hello")
    with pytest.raises(TypeError):
        result.scores["joy"] = 0.0


def test_empty_text_raises():
    analyzer = _analyzer({"joy": 1.0}, StubLexicon(0, 0, 0))
    with pytest.raises(ValueError):
        analyzer.analyze("   ")
