import math

import numpy as np
import pandas as pd
import pytest

from src.stats import safe_pct_gap, weighted_mean, weighted_quantile, weighted_share


class TestWeightedMean:
    def test_uniform_weights(self):
        x = pd.Series([1.0, 2.0, 3.0])
        w = pd.Series([1.0, 1.0, 1.0])
        assert weighted_mean(x, w) == pytest.approx(2.0)

    def test_skewed_weights(self):
        x = pd.Series([0.0, 10.0])
        w = pd.Series([9.0, 1.0])
        assert weighted_mean(x, w) == pytest.approx(1.0)

    def test_all_nan_values_returns_nan(self):
        x = pd.Series([np.nan, np.nan])
        w = pd.Series([1.0, 1.0])
        assert math.isnan(weighted_mean(x, w))

    def test_partial_nan_ignores_missing(self):
        x = pd.Series([np.nan, 4.0])
        w = pd.Series([1.0, 1.0])
        assert weighted_mean(x, w) == pytest.approx(4.0)


class TestWeightedShare:
    def test_two_thirds_eligible(self):
        x = pd.Series([1.0, 1.0, 0.0])
        w = pd.Series([1.0, 1.0, 1.0])
        assert weighted_share(x, w, 1.0) == pytest.approx(2 / 3)

    def test_weighted_share_respects_weights(self):
        x = pd.Series([1.0, 0.0])
        w = pd.Series([3.0, 1.0])
        assert weighted_share(x, w, 1.0) == pytest.approx(0.75)

    def test_all_nan_returns_nan(self):
        x = pd.Series([np.nan, np.nan])
        w = pd.Series([1.0, 1.0])
        assert math.isnan(weighted_share(x, w, 1.0))


class TestWeightedQuantile:
    def test_median_of_five_uniform(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = pd.Series([1.0] * 5)
        assert weighted_quantile(values, weights, 0.5) == pytest.approx(3.0)

    def test_bottom_quantile(self):
        values = pd.Series([10.0, 20.0, 30.0, 40.0])
        weights = pd.Series([1.0] * 4)
        result = weighted_quantile(values, weights, 0.25)
        assert result == pytest.approx(10.0)

    def test_all_nan_returns_nan(self):
        values = pd.Series([np.nan, np.nan])
        weights = pd.Series([1.0, 1.0])
        assert math.isnan(weighted_quantile(values, weights, 0.5))


class TestSafePctGap:
    def test_positive_gap(self):
        assert safe_pct_gap(110.0, 100.0) == pytest.approx(10.0)

    def test_negative_gap(self):
        assert safe_pct_gap(90.0, 100.0) == pytest.approx(-10.0)

    def test_zero_observed_returns_nan(self):
        assert math.isnan(safe_pct_gap(10.0, 0.0))

    def test_nan_simulated_returns_nan(self):
        assert math.isnan(safe_pct_gap(np.nan, 100.0))

    def test_nan_observed_returns_nan(self):
        assert math.isnan(safe_pct_gap(100.0, np.nan))

    def test_equal_values_zero_gap(self):
        assert safe_pct_gap(50.0, 50.0) == pytest.approx(0.0)
