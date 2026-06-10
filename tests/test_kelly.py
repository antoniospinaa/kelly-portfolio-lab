"""Kelly math sanity checks. Single-asset case has a known closed form:
weight = (mu - r) / sigma^2, scaled by the chosen fraction."""
import numpy as np
import pandas as pd
import pytest
from kellyfolio.kelly import kelly_weights


def one_asset(mu_val, var):
    mu = pd.Series({"SPY": mu_val})
    sigma = pd.DataFrame({"SPY": [var]}, index=["SPY"])
    return mu, sigma


def test_single_asset_matches_closed_form():
    mu, sigma = one_asset(0.08, 0.04)
    w = kelly_weights(mu, sigma, risk_free=0.02, fraction=1.0, long_only=False)
    assert w["SPY"] == pytest.approx((0.08 - 0.02) / 0.04)  # = 1.5


def test_weights_scale_linearly_with_fraction():
    mu, sigma = one_asset(0.08, 0.04)
    full = kelly_weights(mu, sigma, risk_free=0.02, fraction=1.0, long_only=False)
    half = kelly_weights(mu, sigma, risk_free=0.02, fraction=0.5, long_only=False)
    assert half["SPY"] == pytest.approx(0.5 * full["SPY"])


def test_long_only_clips_shorts_and_caps_leverage():
    mu = pd.Series({"SPY": 0.10, "TLT": -0.05})
    sigma = pd.DataFrame([[0.04, 0.0], [0.0, 0.02]], index=mu.index, columns=mu.index)
    w = kelly_weights(mu, sigma, risk_free=0.0, fraction=1.0, long_only=True)
    assert (w >= 0).all()
    assert w.sum() <= 1.0 + 1e-9


def test_singular_covariance_does_not_crash():
    mu = pd.Series({"A": 0.05, "B": 0.05})
    sigma = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=mu.index, columns=mu.index)
    w = kelly_weights(mu, sigma, risk_free=0.0, fraction=1.0)
    assert np.isfinite(w).all()
