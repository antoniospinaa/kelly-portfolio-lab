"""Estimate Kelly inputs (mu, Sigma) from historical prices.

KNOWN LIMITATION, ON PURPOSE: expected returns estimated from historical
means are noisy, and Kelly weights are sensitive to that noise. This is
exactly why the project's headline strategy is *fractional* Kelly. v1 keeps
the estimator simple and honest; shrinkage estimators are a roadmap item.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_MONTH = 21


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def estimate_mu_sigma(prices: pd.DataFrame, lookback_months: int = 60):
    """Annualized mean returns and covariance from the last `lookback_months`."""
    returns = log_returns(prices)
    window = lookback_months * DAYS_PER_MONTH
    if len(returns) < window:
        raise ValueError(
            f"Need {window} daily returns for a {lookback_months}-month lookback, "
            f"but only {len(returns)} are available."
        )
    recent = returns.tail(window)
    mu = recent.mean() * TRADING_DAYS
    sigma = recent.cov() * TRADING_DAYS
    return mu, sigma
