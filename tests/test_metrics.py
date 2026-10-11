import math

import pytest

from novavision.eval.metrics import (
    accuracy,
    bootstrap_ci,
    bootstrap_corr_ci,
    cohens_h,
    confusion_matrix,
    holm_bonferroni,
    macro_f1,
    mae,
    majority_baseline,
    paired_bootstrap_test,
    pearson,
    permutation_test,
    prediction_collapse,
    rogan_gladen,
    spearman,
)


def test_mae_known_values():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mae([0.0, 0.0], [1.0, 3.0]) == 2.0


def test_mae_empty_is_nan():
    assert math.isnan(mae([], []))


def test_bootstrap_corr_ci_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_corr_ci([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], method="pearsn")


def test_accuracy():
    assert accuracy(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert math.isnan(accuracy([], []))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a"])


def test_macro_f1_perfect():
    labels = ["a", "b"]
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"], labels) == 1.0


def test_macro_f1_only_present_labels():

    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b", "c"]) == 1.0


def test_confusion_matrix():
    cm = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_out_of_label_raises():
    with pytest.raises(ValueError):
        confusion_matrix(["a", "z"], ["a", "a"], ["a", "b"])


def test_pearson():
    assert round(pearson([1, 2, 3], [2, 4, 6]), 4) == 1.0
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))
    assert math.isnan(pearson([1.0], [2.0]))


def test_spearman_monotonic():
    assert round(spearman([1, 2, 3, 4], [1, 4, 9, 16]), 4) == 1.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    assert lo <= 0.7 <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_paired_test_detects_difference():
    a = [1, 1, 1, 1, 1, 1, 1, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    res = paired_bootstrap_test(a, b)
    assert res["mean_diff"] == 1.0
    assert 0 < res["p_value"] < 0.05


def test_paired_test_no_difference():
    a = [1, 0, 1, 0, 1, 0]
    res = paired_bootstrap_test(a, a)
    assert res["mean_diff"] == 0.0
    assert res["p_value"] >= 0.05


def test_majority_baseline():

    assert majority_baseline(["neutral", "neutral", "neutral", "joy", "anger"]) == 0.6

    assert majority_baseline(["a", "b", "c", "d"]) == 0.25
    assert math.isnan(majority_baseline([]))


def test_prediction_collapse_flags_degenerate_probe():

    c = prediction_collapse(["neutral"] * 9 + ["joy"])
    assert c["label"] == "neutral"
    assert c["rate"] == 0.9
    assert c["distinct"] == 2

    healthy = prediction_collapse(["a", "b", "c", "a", "b", "c"])
    assert round(healthy["rate"], 4) == 0.3333 and healthy["distinct"] == 3


def test_permutation_test_detects_real_signal():
    labels = ["a", "b", "c", "d", "e", "f", "g"] * 4

    perfect = permutation_test(labels, labels, n=500)
    assert perfect["accuracy"] == 1.0
    assert perfect["p_value"] < 0.01
    assert perfect["null_mean"] < 0.3


def test_permutation_test_flags_circular_chance():

    truth = ["a", "b", "c", "d", "e", "f", "g"] * 4
    pred = ["a"] * len(truth)
    res = permutation_test(truth, pred, n=500)
    assert res["p_value"] > 0.2
    assert math.isnan(permutation_test(["a"], ["a"])["p_value"])


def test_bootstrap_corr_ci_brackets_and_widens_at_small_n():

    lo, hi = bootstrap_corr_ci([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    assert -1.0 <= lo <= hi <= 1.0
    assert hi == 1.0

    nlo, nhi = bootstrap_corr_ci([1, 2], [2, 1])
    assert math.isnan(nlo) and math.isnan(nhi)


def test_rogan_gladen_matches_formula_and_bounds():

    assert rogan_gladen(0.3, 1.0, 1.0) == pytest.approx(0.3)

    assert rogan_gladen(0.3, 0.45, 0.9) == pytest.approx(0.2 / 0.35)

    assert rogan_gladen(0.05, 0.9, 0.9) == 0.0

    assert math.isnan(rogan_gladen(0.3, 0.5, 0.5))


def test_cohens_h_sign_and_zero():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.5, 0.2) > 0
    assert cohens_h(0.2, 0.5) < 0


def test_holm_bonferroni_orders_and_rejects():
    adj = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.5}, alpha=0.05)

    assert adj["a"]["reject"] is True
    assert adj["a"]["p_adjusted"] <= adj["b"]["p_adjusted"] <= adj["c"]["p_adjusted"]
    assert adj["c"]["reject"] is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_holm_ignores_non_finite(invalid):
    adj = holm_bonferroni({"a": 0.01, "b": invalid})
    assert adj["a"] == {"p_adjusted": 0.01, "reject": True}
    assert adj["b"]["p_adjusted"] is None and adj["b"]["reject"] is False
