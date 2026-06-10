import numpy as np
import pandas as pd
import pytest
from kellyfolio.backtest import run_backtest, fixed_weights, cagr, max_drawdown


def flat_prices(n_days=300):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame({"SPY": 100.0, "TLT": 50.0}, index=dates)


def test_flat_prices_lose_only_initial_transaction_cost():
    w = pd.Series({"SPY": 0.6, "TLT": 0.4})
    result = run_backtest(flat_prices(), fixed_weights(w), cost=0.001)
    # One initial buy of 100% of the portfolio costs 0.1%; afterwards
    # weights never drift on flat prices, so no further trades.
    assert result.equity.iloc[-1] == pytest.approx(1 - 0.001, rel=1e-6)


def test_strategy_never_sees_the_future():
    seen = []

    def spy_fn(history):
        seen.append(history.index[-1])
        return pd.Series({"SPY": 1.0, "TLT": 0.0})

    prices = flat_prices()
    result = run_backtest(prices, spy_fn, cost=0.0)
    rebalance_dates = result.weights.index
    assert list(seen)[: len(rebalance_dates)] == list(rebalance_dates)


def test_warmup_skipped_until_weight_fn_ready():
    def needs_history(history):
        if len(history) < 100:
            return None  # "not enough data yet" — engine must skip, not crash
        return pd.Series({"SPY": 1.0, "TLT": 0.0})

    result = run_backtest(flat_prices(), needs_history, cost=0.0)
    assert len(result.weights) > 0


def test_metrics():
    dates = pd.bdate_range("2020-01-01", periods=2)
    doubled = pd.Series([1.0, 2.0], index=[dates[0], dates[0] + pd.Timedelta(days=365)])
    assert cagr(doubled) == pytest.approx(1.0, rel=0.01)
    dipped = pd.Series([1.0, 0.5, 0.8])
    assert max_drawdown(dipped) == pytest.approx(-0.5)
