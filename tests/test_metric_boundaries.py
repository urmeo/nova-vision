"""Reject incomplete pairs and avoid invented correlations for missing observations."""

import math

import pytest

from novavision.eval.metrics import bootstrap_corr_ci, mae, pearson, permutation_test, spearman


@pytest.mark.parametrize("metric", [mae, pearson, spearman, bootstrap_corr_ci])
@pytest.mark.parametrize("x,y", [([0.1, 0.2, 0.3], [0.1]), ([0.1], [0.1, 0.2, 0.3]), ([], [0.1])])
def test_numeric_metrics_reject_unpaired_observations(metric, x, y):
    with pytest.raises(ValueError, match="same length"):
        metric(x, y)


@pytest.mark.parametrize(
    "truth,predicted",
    [(["joy", "sadness", "fear"], ["joy"]), (["joy"], ["joy", "fear"]), ([], ["joy"])],
)
def test_permutation_test_rejects_unpaired_labels(truth, predicted):
    with pytest.raises(ValueError, match="same length"):
        permutation_test(truth, predicted, n=10)


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("missing_side", ["x", "y"])
def test_rank_correlation_is_unavailable_with_nonfinite_observations(missing, missing_side):
    x, y = [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]
    (x if missing_side == "x" else y)[1] = missing
    assert math.isnan(spearman(x, y))
    assert all(math.isnan(bound) for bound in bootstrap_corr_ci(x, y, n=10))
