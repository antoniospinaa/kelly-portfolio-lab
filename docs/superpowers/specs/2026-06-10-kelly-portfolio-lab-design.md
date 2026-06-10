# Kelly Portfolio Lab — Design

**Date:** 2026-06-10
**Status:** Approved by Tony (brainstorming session)
**Source:** Capstone — "Maximizing Long-Term Investment Growth: A Study of the Kelly Criterion's Portfolio Applications"

## Purpose

Turn Tony's Kelly Criterion capstone into a GitHub portfolio project aimed at **data/quant analyst** roles, and into a practical tool he can use to allocate his own ETF investments.

Success criteria:
- A recruiter or hiring manager can open the repo and, within minutes, see the methodology, the results, and clean tested code.
- Tony (Python beginner) can explain every line in an interview.
- The CLI report gives a real monthly allocation he can execute manually in his broker.

## Scope

**Phase 1 (this design):** paper-only. Historical backtests plus a "today's allocation" report. No broker connection, no automated trading.

**Phase 2 (out of scope):** Streamlit dashboard demo.
**Phase 3 (out of scope):** Alpaca paper-trading integration; real money only after the demo proves out.

Asset universe: 3–6 ETFs (e.g., broad equity, bonds, gold) defined in `config.yaml`.

## Architecture

```
kelly-portfolio-lab/
├── README.md              ← results, charts, link to capstone PDF
├── config.yaml            ← tickers, risk-free rate, Kelly fraction, lookback, costs
├── src/kellyfolio/
│   ├── data.py            ← download & cache ETF prices (yfinance)
│   ├── estimate.py        ← expected returns vector μ + covariance matrix Σ
│   ├── kelly.py           ← full & fractional Kelly weights
│   ├── backtest.py        ← walk-forward backtest + metrics
│   └── report.py          ← today's suggested allocation (CLI entry point)
├── notebooks/
│   └── kelly_story.ipynb  ← capstone narrative with live charts
├── tests/
└── .github/workflows/ci.yml
```

Each module is one idea, target 50–100 lines, so a beginner can own it.

### Data flow

`config.yaml` → `data.py` (download/cache prices) → `estimate.py` (μ, Σ from log returns) → `kelly.py` (weights = f · Σ⁻¹(μ − r), with f the Kelly fraction) → consumed by either `backtest.py` (historical replay) or `report.py` (current allocation).

### Module contracts

- **data.py** — `get_prices(tickers, start) -> DataFrame` (adjusted close, business-day index). Caches to a local parquet/CSV; on network failure falls back to cache with a warning.
- **estimate.py** — `estimate_mu_sigma(prices, lookback_months) -> (Series, DataFrame)`. Simple historical means and sample covariance, annualized. Shrinkage estimators are a roadmap item, deliberately not v1.
- **kelly.py** — `kelly_weights(mu, sigma, risk_free, fraction) -> Series`. Continuous-time multi-asset Kelly: w = fraction × Σ⁻¹(μ − r). Optional constraint: no leverage / no shorting (clip and renormalize), on by default for realism.
- **backtest.py** — `run_backtest(prices, config) -> BacktestResult` with equity curve, CAGR, max drawdown, Sharpe, per-rebalance weights. Monthly rebalance.
- **report.py** — `python -m kellyfolio report` prints current target weights and, given a portfolio value, dollar amounts per ETF.

## Methodology decisions (the quant-credibility core)

1. **Walk-forward, no look-ahead.** At each monthly rebalance, μ and Σ are estimated only from data available up to that date.
2. **Fractional Kelly headline.** Compare full, ½, ¼ Kelly vs buy-and-hold S&P 500 and 60/40. Growth curves and drawdowns side by side.
3. **Honest estimation error.** v1 uses historical means with a long lookback and says so explicitly in the README; named limitation, not hidden.
4. **Transaction costs.** Flat per-trade cost (default 0.1% of turnover) applied at each rebalance.

## Error handling

- Unknown ticker → clear message naming the ticker.
- Network failure → use cached prices with a staleness warning; fail clearly if no cache.
- Insufficient history for the lookback window → clear message stating required vs available.
- Singular/near-singular Σ (e.g., duplicated tickers) → ridge fallback with warning.

## Testing

Test the math, not the network (fixtures with synthetic prices):

- Single asset with known μ, σ²: Kelly weight equals (μ − r)/σ².
- Weights scale linearly with the Kelly fraction.
- Flat-price backtest → zero growth, costs only.
- No-look-ahead check: estimates at date t unchanged when future data is appended.
- Constraint behavior: no-shorting clip produces non-negative weights summing ≤ 1.

GitHub Actions runs pytest on every push; badge in README.

## README (a deliverable, not an afterthought)

Sections: what the Kelly Criterion is (3 sentences), headline backtest chart, methodology summary (the 4 decisions above), how to run, limitations, roadmap (Phases 2–3), link to the capstone PDF.

## Tech

Python 3.12, `numpy`, `pandas`, `yfinance`, `matplotlib`, `pyyaml`, `pytest`. Package layout under `src/`, installable with `pip install -e .`. No frameworks beyond that — beginner-ownable is a hard constraint.
