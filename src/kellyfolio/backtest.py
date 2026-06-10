"""Walk-forward monthly backtest.

No look-ahead by construction: at each month-end the strategy receives only
prices up to that date, chooses weights, and holds them for the next month.
A flat transaction cost is charged on turnover at every rebalance.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series      # portfolio value over time, starts at 1.0
    weights: pd.DataFrame  # chosen weights at each rebalance date


def month_ends(prices: pd.DataFrame) -> list:
    """Last trading day of each month in the data."""
    return list(prices.groupby(prices.index.to_period("M")).tail(1).index)


def run_backtest(prices: pd.DataFrame, weight_fn, cost: float = 0.001) -> BacktestResult:
    dates = month_ends(prices)
    equity, prev_w = 1.0, pd.Series(0.0, index=prices.columns)
    equity_curve, weight_rows = {}, {}

    for t0, t1 in zip(dates[:-1], dates[1:]):
        w = weight_fn(prices.loc[:t0])
        if w is None:  # strategy not ready (warm-up period)
            continue
        w = w.reindex(prices.columns).fillna(0.0)
        equity *= 1 - cost * (w - prev_w).abs().sum()
        period_return = prices.loc[t1] / prices.loc[t0] - 1
        equity *= 1 + (w * period_return).sum()  # unallocated weight sits in cash at 0%
        equity_curve[t1], weight_rows[t0], prev_w = equity, w, w

    return BacktestResult(pd.Series(equity_curve), pd.DataFrame(weight_rows).T)


def fixed_weights(weights: pd.Series):
    """A benchmark strategy that rebalances to the same weights every month."""
    return lambda history: weights


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    return (equity / equity.cummax() - 1).min()


def sharpe(equity: pd.Series, risk_free: float = 0.0) -> float:
    monthly = equity.pct_change().dropna()
    excess = monthly - risk_free / 12
    return float(excess.mean() / excess.std() * np.sqrt(12))
