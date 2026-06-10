"""Estimation tests use synthetic prices with known growth rates."""
import numpy as np
import pandas as pd
import pytest
from kellyfolio.estimate import estimate_mu_sigma, TRADING_DAYS


def synthetic_prices(daily_log_return, n_days=2000):
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    levels = np.exp(np.arange(n_days) * daily_log_return)
    return pd.DataFrame({"SPY": levels}, index=dates)


def test_constant_growth_recovers_annualized_mean():
    prices = synthetic_prices(0.001)
    mu, sigma = estimate_mu_sigma(prices, lookback_months=24)
    assert mu["SPY"] == pytest.approx(0.001 * TRADING_DAYS)
    assert sigma.loc["SPY", "SPY"] == pytest.approx(0.0, abs=1e-12)


def test_insufficient_history_raises_clear_error():
    prices = synthetic_prices(0.001, n_days=100)
    with pytest.raises(ValueError, match="lookback"):
        estimate_mu_sigma(prices, lookback_months=24)


def test_estimates_use_only_the_lookback_tail():
    # First half crazy growth, second half flat: a 12-month lookback
    # must reflect only the flat recent regime.
    early = synthetic_prices(0.01, n_days=1000)
    late_levels = early["SPY"].iloc[-1] * np.ones(1000)
    late = pd.DataFrame({"SPY": late_levels},
                        index=pd.bdate_range(early.index[-1] + pd.Timedelta(days=1), periods=1000))
    prices = pd.concat([early, late])
    mu, _ = estimate_mu_sigma(prices, lookback_months=12)
    assert mu["SPY"] == pytest.approx(0.0, abs=1e-9)
