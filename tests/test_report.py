import pytest
import report


def _results():
    return {
        "metrics": {
            "raw": {
                "accuracy": 0.14,
                "accuracy_ci": [0.1, 0.2],
                "macro_f1": 0.1,
                "valence_rho": 0.0,
                "arousal_rho": 0.0,
                "clip_t": 0.2,
                "n": 70,
            },
            "affect": {
                "accuracy": 0.55,
                "accuracy_ci": [0.45, 0.65],
                "macro_f1": 0.5,
                "valence_rho": 0.3,
                "arousal_rho": 0.2,
                "clip_t": 0.25,
                "n": 70,
            },
            "chance": 0.143,
        },
        "contrasts": {
            "affect_vs_raw": {"mean_diff": 0.41, "ci_low": 0.3, "ci_high": 0.5, "p_value": 0.001}
        },
    }


def test_render_includes_ci_and_significance():
    out = report.render(_results())
    assert "[0.450, 0.650]" in out
    assert "+0.410" in out
    assert "chance" in out.lower()


def test_nan_renders_as_placeholder():
    assert report._fmt(float("nan")) == "n/a"
    assert report._fmt(None) == "n/a"


@pytest.mark.parametrize(
    "bounds", [[0.1, None], [None, 0.9], [0.1, float("nan")], [0.1, float("inf")]]
)
def test_rho_omits_incomplete_or_nonfinite_interval(bounds):
    metrics = {"valence_rho": 0.5, "valence_rho_ci": bounds}
    assert report._rho(metrics, "valence_rho") == "0.500"


def test_shuffled_note_survives_partial_control_dict():
    metrics = {
        "emotion": {"shuffled_control": {}},
        "affect": {"shuffled_control": {"p_value": 0.14, "null_mean": 0.142}},
    }
    note = report._shuffled_note(metrics)
    assert "affect p=0.14" in note
    assert "null mean 0.142" in note


def test_shuffled_note_empty_when_no_pvalues():
    assert report._shuffled_note({"emotion": {"shuffled_control": {"null_mean": 0.1}}}) == ""


def test_tables_tolerate_null_bounds():
    metrics = {
        "raw": {
            "accuracy": 0.14,
            "accuracy_ci": [None, None],
            "macro_f1": None,
            "valence_rho": 0.0,
            "arousal_rho": 0.0,
            "clip_t": None,
            "n": 1,
        },
        "chance": 0.143,
    }
    assert "n/a" in report.metrics_table(metrics)
    contrasts = {
        "emotion_vs_raw": {"ci_low": None, "ci_high": None, "mean_diff": None, "p_value": None}
    }
    assert "n/a" in report.contrasts_table(contrasts)


def test_rho_tolerates_null_ci():
    assert (
        report._rho({"valence_rho": None, "valence_rho_ci": [None, None]}, "valence_rho") == "n/a"
    )
    assert "[" in report._rho({"valence_rho": 0.5, "valence_rho_ci": [0.1, 0.9]}, "valence_rho")
