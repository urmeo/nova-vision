import pytest

from novavision.affect.lexicon import AffectLexicon


@pytest.fixture(scope="module")
def lex():
    return AffectLexicon.load()


def test_loads_entries(lex):
    assert len(lex) > 50


def test_positive_text_has_positive_valence(lex):
    assert lex.score("I feel so happy and joyful today").valence > 0.3


def test_negative_text_has_negative_valence(lex):
    assert lex.score("I am sad lonely and hopeless").valence < -0.3


def test_load_strips_field_whitespace(tmp_path):

    f = tmp_path / "lex.tsv"
    f.write_text("word \tvalence\tarousal\nzephyrword \t 0.8 \t 0.6 \n", encoding="utf-8")
    s = AffectLexicon.load(f).score("zephyrword")
    assert s.coverage == 1.0 and s.valence == 0.8


def test_anger_has_high_arousal(lex):
    assert lex.score("furious rage outraged").arousal > 0.7


def test_empty_text_is_neutral(lex):
    score = lex.score("")
    assert score.coverage == 0.0
    assert score.valence == 0.0


def test_coverage_bounds(lex):
    score = lex.score("the happy dog ran quickly")
    assert 0.0 < score.coverage <= 1.0


def test_suffix_variant_lookup(lex):
    assert lex.lookup("loves") == lex.lookup("love")


def test_inflections_restore_spelling():
    lex = AffectLexicon({"love": (0.9, 0.7), "care": (0.6, 0.4), "run": (0.1, 0.8)})
    assert lex.lookup("loving") == (0.9, 0.7)
    assert lex.lookup("caring") == (0.6, 0.4)
    assert lex.lookup("running") == (0.1, 0.8)


def test_ies_of_ie_nouns():
    lex = AffectLexicon({"movie": (0.3, 0.5), "city": (0.1, 0.4)})
    assert lex.lookup("movies") == (0.3, 0.5)
    assert lex.lookup("cities") == (0.1, 0.4)


def test_meaning_changing_suffixes_not_stripped():

    lex = AffectLexicon({"hon": (0.5, 0.5), "inter": (0.0, 0.5), "hope": (0.8, 0.5)})
    assert lex.lookup("honest") is None
    assert lex.lookup("interest") is None
    assert lex.lookup("hopeless") is None


def test_ly_suffix_not_stripped():

    lex = AffectLexicon({"love": (0.9, 0.7), "ear": (0.0, 0.4)})
    assert lex.lookup("lovely") is None
    assert lex.lookup("early") is None


def test_malformed_lexicon_line_raises(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("word\tvalence\tarousal\nhappy\thigh\t0.5\n")
    with pytest.raises(ValueError):
        AffectLexicon.load(bad)


@pytest.mark.parametrize(
    "values", [("nan", "0.5"), ("0.5", "inf"), ("1.1", "0.5"), ("0.5", "-0.1")]
)
def test_invalid_norms_are_rejected_before_scoring(tmp_path, values):
    bad = tmp_path / "bad.tsv"
    bad.write_text(f"word\tvalence\tarousal\nhappy\t{values[0]}\t{values[1]}\n")
    with pytest.raises(ValueError, match="Invalid affect"):
        AffectLexicon.load(bad)


def test_caller_cannot_change_loaded_norms():
    entries = {"happy": (0.8, 0.7)}
    lexicon = AffectLexicon(entries)
    entries["happy"] = (float("nan"), 2.0)
    assert lexicon.score("happy").valence == 0.8


def test_negation_flips_valence():
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score("not happy").valence == -0.8
    assert lex.score("i am never happy").valence == -0.8
    assert lex.score("happy").valence == 0.8


@pytest.mark.parametrize(
    "boundary",
    [
        ".",
        "!",
        "?",
        ",",
        ";",
        ":",
        "\n",
        "\r\n",
        "—",
        "–",
        " -- ",
        " - ",
    ],
)
def test_negation_stops_at_punctuation_boundaries(boundary):
    lex = AffectLexicon({"sad": (-0.6, 0.3), "happy": (0.8, 0.7)})
    score = lex.score(f"Not sad{boundary} Happy!")
    assert score.valence == 0.7
    assert score.arousal == 0.5
    assert score.coverage == 1.0
    assert lex.score(f"Sad{boundary} Not happy!").valence == -0.7


@pytest.mark.parametrize("contrast", ["but", "however", "yet"])
def test_negation_stops_at_contrast_clauses(contrast):
    lex = AffectLexicon({"sad": (-0.6, 0.3), "happy": (0.8, 0.7)})
    score = lex.score(f"not sad {contrast} happy")
    assert score.valence == 0.7
    assert score.arousal == 0.5
    assert score.coverage == (1.0 if contrast == "but" else 0.6667)
    assert lex.score(f"not sad {contrast} not happy").valence == -0.1


@pytest.mark.parametrize(
    "text",
    [
        "not very happy",
        "not yet happy",
        "not-happy",
        "not 'happy'",
        "not (happy)",
        "not (really) happy",
    ],
)
def test_same_clause_negation_keeps_two_word_window(text):
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score(text).valence == -0.8


def test_negation_does_not_extend_beyond_two_words():
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score("not really very happy").valence == 0.8


def test_stopwords_are_not_scored_or_counted():
    lex = AffectLexicon({"happy": (0.8, 0.7), "the": (0.9, 0.9)})
    s = lex.score("the happy dog")
    assert s.valence == 0.8
    assert s.coverage == 0.5


def test_smart_quote_negation_still_flips():
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score("i can’t be happy").valence == -0.8


@pytest.mark.parametrize("text", ["'happy'", "‘happy’", '"happy"'])
def test_quoted_words_are_scored(text):
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score(text) == lex.score("happy")


@pytest.mark.parametrize("text", ["not 'happy'", "not ‘happy’", "I don't feel 'happy'"])
def test_negation_survives_quoted_words(text):
    lex = AffectLexicon({"happy": (0.8, 0.7)})
    assert lex.score(text).valence == -0.8


def test_load_tolerates_utf8_bom(tmp_path):
    path = tmp_path / "lexicon.tsv"
    path.write_text("word\tvalence\tarousal\nhappy\t0.8\t0.7\n", encoding="utf-8-sig")
    assert AffectLexicon.load(path).lookup("happy") == (0.8, 0.7)


def test_inflection_cannot_stem_into_a_function_word():
    lex = AffectLexicon({"will": (0.5, 0.5), "happy": (0.8, 0.7)})
    assert lex.lookup("wills") is None
