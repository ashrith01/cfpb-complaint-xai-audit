"""PSI arithmetic and binning. The drift report is only as trustworthy as these."""

import numpy as np
import pytest

from src.monitor import TokenBins, binned, categorical, psi, quantile_edges, verdict


def test_identical_distributions_have_zero_psi():
    assert psi(np.array([10, 20, 30]), np.array([1, 2, 3])) == pytest.approx(0.0)


def test_psi_is_symmetric_and_positive():
    a, b = np.array([50, 30, 20]), np.array([20, 30, 50])
    assert psi(a, b) > 0
    assert psi(a, b) == pytest.approx(psi(b, a))


def test_known_value():
    # (0.5-0.25)ln(2) + (0.25-0.5)ln(0.5) + 0 = 0.5 ln 2
    assert psi(np.array([1, 2, 1]), np.array([2, 1, 1])) == pytest.approx(0.5 * np.log(2))


def test_empty_bin_is_finite():
    assert np.isfinite(psi(np.array([10, 0]), np.array([5, 5])))


def test_verdict_bands():
    assert verdict(0.05) == "stable"
    assert verdict(0.1) == "moderate"
    assert verdict(0.3) == "significant"


def test_quantile_edges_capture_out_of_range_values():
    edges = quantile_edges(np.arange(100))
    counts = binned(np.array([-1e9, 50, 1e9]), edges)
    assert counts.sum() == 3


def test_categorical_counts_absent_classes_as_zero():
    assert categorical([0, 0, 2], [0, 1, 2]).tolist() == [2, 0, 1]


def test_unseen_tokens_go_to_other_bin():
    bins = TokenBins(["credit card credit"], n=2)
    counts = bins.counts(["credit mortgage mortgage"])
    assert counts[-1] == 2 and counts.sum() == 3
