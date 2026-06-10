import pandas as pd
from kellyfolio.report import kelly_strategy, format_report


def test_format_report_shows_percent_and_dollars():
    w = pd.Series({"SPY": 0.45, "TLT": 0.30})
    text = format_report(w, portfolio_value=10_000)
    assert "SPY" in text and "45.0%" in text and "$4,500" in text
    assert "Cash" in text and "25.0%" in text  # 1 - 0.45 - 0.30


def test_kelly_strategy_returns_none_during_warmup():
    config = {"risk_free_rate": 0.02, "kelly_fraction": 0.5, "lookback_months": 60}
    short_history = pd.DataFrame(
        {"SPY": [100.0, 101.0]}, index=pd.bdate_range("2020-01-01", periods=2))
    assert kelly_strategy(config)(short_history) is None
