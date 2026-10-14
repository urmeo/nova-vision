import itertools
import math

import numpy as np
import pytest

from novavision.eval.metrics import mcnemar_exact, paired_binary_test, paired_bootstrap_test


def _exhaustive_p_value(a, b):
    differences = [x - y for x, y in zip(a, b) if x != y]
    observed = abs(sum(differences))
    outcomes = list(itertools.product((-1, 1), repeat=len(differences)))
    return sum(abs(sum(outcome)) >= observed for outcome in outcomes) / len(outcomes)


def test_two_discordant_pairs_cannot_establish_significance():
    result = paired_binary_test([1, 1], [0, 0])

    assert result["mean_diff"] == 1.0
    assert result["p_value"] == 0.5


def test_previous_paired_test_api_uses_corrected_significance():
    result = paired_bootstrap_test([1, 1], [0, 0])

    assert result == paired_binary_test([1, 1], [0, 0])
    assert result["p_value"] == 0.5


@pytest.mark.parametrize("a,b", [([1], [0]), ([0], [1]), ([1], [1]), ([0], [0])])
def test_one_pair_has_exact_p_value_one(a, b):
    result = paired_binary_test(a, b)

    assert result["mean_diff"] == a[0] - b[0]
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])
    assert result["p_value"] == 1.0


@pytest.mark.parametrize("discordant", range(1, 9))
def test_exact_test_matches_exhaustive_paired_randomization(discordant):
    for wins in range(discordant + 1):
        a = [1] * wins + [0] * (discordant - wins)
        b = [1 - value for value in a]

        result = paired_binary_test(a, b, n=32)

        assert result["mean_diff"] == pytest.approx(sum(a) / len(a) - sum(b) / len(b))
        assert result["p_value"] == _exhaustive_p_value(a, b)
        exact = mcnemar_exact(a, b)
        assert exact["only_a_correct"] == wins
        assert exact["only_b_correct"] == discordant - wins
        assert exact["p_value"] == result["p_value"]


@pytest.mark.parametrize("a,b", [([1, 1], [0, 0]), ([1, 0, 1], [0, 1, 0])])
def test_tied_pairs_do_not_change_exact_p_value(a, b):
    result = paired_binary_test(a, b)
    with_ties = paired_binary_test(a + [0, 1, 1, 0], b + [0, 1, 1, 0])

    assert with_ties["p_value"] == result["p_value"]
    assert with_ties["mean_diff"] == pytest.approx(result["mean_diff"] * len(a) / (len(a) + 4))


def test_all_tied_pairs_have_zero_difference_and_p_value_one():
    result = paired_binary_test([1, 0, 1, 1], [1, 0, 1, 1])

    assert result["mean_diff"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert result["p_value"] == 1.0


def test_swapping_models_reverses_difference_but_not_significance():
    a = [1, 1, 1, 0, 1, 0]
    b = [0, 0, 1, 1, 0, 0]
    forward = paired_binary_test(a, b)
    reverse = paired_binary_test(b, a)

    assert reverse["mean_diff"] == -forward["mean_diff"]
    assert reverse["ci_low"] == -forward["ci_high"]
    assert reverse["ci_high"] == -forward["ci_low"]
    assert reverse["p_value"] == forward["p_value"]


def test_exact_p_value_does_not_depend_on_bootstrap_seed_or_count():
    a = [1, 1, 1, 1, 0, 1, 0]
    b = [0, 0, 0, 1, 1, 0, 0]
    results = [paired_binary_test(a, b, n=n, seed=seed) for n, seed in [(10, 0), (99, 5)]]

    assert results[0]["p_value"] == results[1]["p_value"] == _exhaustive_p_value(a, b)


def test_boolean_correctness_is_supported():
    result = paired_binary_test([True, True, False], [False, False, False])

    assert result["mean_diff"] == pytest.approx(2 / 3)
    assert result["p_value"] == 0.5


def test_empty_pairs_are_unmeasured():
    result = paired_binary_test([], [])

    assert set(result) == {"mean_diff", "ci_low", "ci_high", "p_value"}
    for key in ("mean_diff", "ci_low", "ci_high", "p_value"):
        assert math.isnan(result[key])


def test_exact_test_with_no_discordance_has_p_value_one():
    assert mcnemar_exact([], []) == {
        "only_a_correct": 0,
        "only_b_correct": 0,
        "p_value": 1.0,
    }


def test_exact_test_preserves_representable_small_probabilities():
    result = mcnemar_exact([1] * 1024, [0] * 1024)

    assert result["only_a_correct"] == 1024
    assert result["only_b_correct"] == 0
    assert result["p_value"] == 2.0**-1023


def test_exact_test_handles_large_one_sided_counts_without_overflow():
    result = mcnemar_exact([1] * 1100, [0] * 1100)

    assert result["only_a_correct"] == 1100
    assert result["only_b_correct"] == 0
    assert result["p_value"] == 0.0


def test_exact_test_handles_large_balanced_counts_without_overflow():
    result = mcnemar_exact([1] * 550 + [0] * 550, [0] * 550 + [1] * 550)

    assert result == {"only_a_correct": 550, "only_b_correct": 550, "p_value": 1.0}


@pytest.mark.parametrize("a,b", [([1, 0], [1]), ([], [0])])
def test_mismatched_pairs_are_rejected(a, b):
    with pytest.raises(ValueError):
        paired_binary_test(a, b)


@pytest.mark.parametrize("invalid", [[0, 0.5], [0, 2], [0, -1], [0, np.nan], [0, np.inf]])
@pytest.mark.parametrize("side", ["a", "b"])
def test_non_binary_or_non_finite_correctness_is_rejected(invalid, side):
    a, b = (invalid, [0, 1]) if side == "a" else ([0, 1], invalid)

    with pytest.raises(ValueError):
        paired_binary_test(a, b)


@pytest.mark.parametrize("invalid", [1, [[0, 1]], np.zeros((2, 1)), [[0], [1, 0]]])
@pytest.mark.parametrize("side", ["a", "b"])
def test_correctness_must_be_one_dimensional(invalid, side):
    a, b = (invalid, [0, 1]) if side == "a" else ([0, 1], invalid)

    with pytest.raises(ValueError):
        paired_binary_test(a, b)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, None, "10"])
def test_bootstrap_count_must_be_a_positive_integer(invalid):
    with pytest.raises(ValueError):
        paired_binary_test([1, 0], [0, 1], n=invalid)
