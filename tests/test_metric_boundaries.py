"""Reject incomplete pairs and avoid invented correlations for missing observations."""

import math

import numpy as np
import pytest

from novavision.eval.metrics import (
    bootstrap_ci,
    bootstrap_corr_ci,
    holm_bonferroni,
    mae,
    pearson,
    permutation_test,
    spearman,
)


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


def test_permutation_p_value_keeps_precision_for_significance(monkeypatch):
    class Shuffles:
        calls = 0

        def permutation(self, values):
            self.calls += 1
            return values if self.calls <= 1000 else values[::-1]

    monkeypatch.setattr(np.random, "default_rng", lambda seed: Shuffles())
    result = permutation_test(["joy", "sadness"], ["joy", "sadness"], n=20000)

    assert result["p_value"] == 1001 / 20001
    assert result["p_value"] > 0.05
    assert not holm_bonferroni({"test": result["p_value"]})["test"]["reject"]


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
@pytest.mark.parametrize("metric", [bootstrap_ci, bootstrap_corr_ci, permutation_test])
def test_resampling_rejects_invalid_counts(metric, n):
    args = ([0.1, 0.2, 0.3],) if metric is bootstrap_ci else ([0.1, 0.2, 0.3],) * 2
    with pytest.raises(ValueError, match="positive integer"):
        metric(*args, n=n)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("metric", [bootstrap_ci, bootstrap_corr_ci])
def test_bootstrap_rejects_invalid_confidence_level(metric, alpha):
    args = ([0.1, 0.2, 0.3],) if metric is bootstrap_ci else ([0.1, 0.2, 0.3],) * 2
    with pytest.raises(ValueError, match="alpha"):
        metric(*args, alpha=alpha)
