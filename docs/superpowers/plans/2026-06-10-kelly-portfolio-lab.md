# Kelly Portfolio Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small, readable Python package that computes fractional-Kelly ETF allocations, backtests them walk-forward (no look-ahead) against buy-and-hold SPY and a 60/40 portfolio, and prints today's suggested allocation.

**Architecture:** Five focused modules under `src/kellyfolio/` (data → estimate → kelly → backtest → report), each under ~100 lines. The backtest engine takes a `weight_fn(history)` callback that only ever receives prices up to the rebalance date — no look-ahead is enforced by construction, and the same engine runs both the Kelly strategy and the fixed-weight benchmarks (DRY).

**Tech Stack:** Python 3.12, numpy, pandas, yfinance, matplotlib, pyyaml, pytest. No frameworks.

**Spec:** `docs/superpowers/specs/2026-06-10-kelly-portfolio-lab-design.md`

**Hard constraints (from the user, for recruiter appeal):**
1. Readable code, every file under ~100 lines, no heavy frameworks.
2. No cheating: walk-forward backtest uses only past data.
3. Own the flaws: README states estimation-error limitations explicitly.
4. Protect the downside: fractional Kelly (½, ¼) compared against 60/40 and buy-and-hold SPY.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config.yaml`, `src/kellyfolio/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "kellyfolio"
version = "0.1.0"
description = "Fractional-Kelly ETF portfolio allocator and walk-forward backtester"
requires-python = ">=3.11"
dependencies = ["numpy", "pandas", "yfinance", "matplotlib", "pyyaml"]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.egg-info/
.venv/
data/
.ipynb_checkpoints/
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
tickers: [SPY, TLT, GLD]
start_date: "2006-01-01"
risk_free_rate: 0.02      # annual, used as the Kelly hurdle
kelly_fraction: 0.5       # half-Kelly: the headline strategy
lookback_months: 60       # history window for estimating mu and sigma
transaction_cost: 0.001   # 0.1% of turnover per rebalance
```

- [ ] **Step 4: Create empty `src/kellyfolio/__init__.py` and `tests/__init__.py`**

- [ ] **Step 5: Create venv and install**

Run: `cd ~/Code-projects/kelly-portfolio-lab && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
Expected: installs without error. Run `.venv/bin/pytest` → "no tests ran" (exit code 5 is fine).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold kellyfolio package"
```

---

### Task 2: `kelly.py` — the core math

**Files:**
- Create: `src/kellyfolio/kelly.py`
- Test: `tests/test_kelly.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_kelly.py -v`
Expected: FAIL / error — `No module named 'kellyfolio.kelly'`

- [ ] **Step 3: Implement `src/kellyfolio/kelly.py`**

```python
"""Kelly Criterion portfolio weights.

Continuous-time multi-asset Kelly: w = fraction * inverse(Sigma) @ (mu - r).
Full Kelly (fraction=1) maximizes long-run log growth but is famously too
aggressive in practice because mu is estimated with error; fractional Kelly
trades a little growth for much smaller drawdowns.
"""
import numpy as np
import pandas as pd

RIDGE = 1e-8  # tiny diagonal bump so a singular covariance matrix still solves


def kelly_weights(mu: pd.Series, sigma: pd.DataFrame, risk_free: float = 0.0,
                  fraction: float = 1.0, long_only: bool = True) -> pd.Series:
    """Return target portfolio weights, indexed like `mu`.

    long_only=True reflects a normal brokerage account: negative weights are
    clipped to zero and, if the total exceeds 100%, weights are scaled down
    so the portfolio never uses leverage. Leftover weight is implicit cash.
    """
    excess = (mu - risk_free).values
    cov = sigma.values + RIDGE * np.eye(len(mu))
    raw = np.linalg.solve(cov, excess)
    weights = pd.Series(fraction * raw, index=mu.index)

    if long_only:
        weights = weights.clip(lower=0.0)
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
    return weights
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_kelly.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/kellyfolio/kelly.py tests/test_kelly.py
git commit -m "feat: Kelly weights with fractional scaling and long-only constraint"
```

---

### Task 3: `estimate.py` — turning prices into μ and Σ

**Files:**
- Create: `src/kellyfolio/estimate.py`
- Test: `tests/test_estimate.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_estimate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/kellyfolio/estimate.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_estimate.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/kellyfolio/estimate.py tests/test_estimate.py
git commit -m "feat: annualized mu/sigma estimation with lookback window"
```

---

### Task 4: `data.py` — prices with cache fallback

**Files:**
- Create: `src/kellyfolio/data.py`
- Test: `tests/test_data.py`

Design note: the downloader is injected as a function argument so tests never
touch the network. Default is yfinance.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pytest
from kellyfolio.data import get_prices


def fake_prices():
    dates = pd.bdate_range("2020-01-01", periods=10)
    return pd.DataFrame({"SPY": range(100, 110), "TLT": range(50, 60)},
                        index=dates, dtype=float)


def test_download_saves_cache(tmp_path):
    cache = tmp_path / "prices.csv"
    out = get_prices(["SPY", "TLT"], "2020-01-01", cache_path=cache,
                     downloader=lambda t, s: fake_prices())
    assert cache.exists()
    assert list(out.columns) == ["SPY", "TLT"]


def test_network_failure_falls_back_to_cache(tmp_path):
    cache = tmp_path / "prices.csv"

    def boom(t, s):
        raise ConnectionError("no internet")

    get_prices(["SPY", "TLT"], "2020-01-01", cache_path=cache,
               downloader=lambda t, s: fake_prices())
    with pytest.warns(UserWarning, match="cached"):
        out = get_prices(["SPY", "TLT"], "2020-01-01", cache_path=cache, downloader=boom)
    assert len(out) == 10


def test_network_failure_without_cache_raises(tmp_path):
    def boom(t, s):
        raise ConnectionError("no internet")

    with pytest.raises(ConnectionError):
        get_prices(["SPY"], "2020-01-01", cache_path=tmp_path / "none.csv", downloader=boom)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/kellyfolio/data.py`**

```python
"""Download adjusted ETF prices, with a local CSV cache as offline fallback."""
import warnings
from pathlib import Path

import pandas as pd


def _yf_download(tickers, start):
    import yfinance as yf
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):  # yfinance returns a Series for one ticker
        data = data.to_frame(tickers[0])
    return data.dropna()


def get_prices(tickers, start, cache_path="data/prices.csv", downloader=_yf_download):
    """Daily adjusted close prices, one column per ticker."""
    cache_path = Path(cache_path)
    try:
        prices = downloader(list(tickers), start)
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"No price data for ticker(s): {missing}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        return prices
    except (ValueError,):
        raise
    except Exception:
        if cache_path.exists():
            warnings.warn(f"Download failed; using cached prices from {cache_path}. "
                          "Data may be stale.")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        raise
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/kellyfolio/data.py tests/test_data.py
git commit -m "feat: price download with offline cache fallback"
```

---

### Task 5: `backtest.py` — walk-forward engine + metrics

**Files:**
- Create: `src/kellyfolio/backtest.py`
- Test: `tests/test_backtest.py`

Design note: `run_backtest(prices, weight_fn, cost)` calls
`weight_fn(prices.loc[:rebalance_date])` — the strategy physically cannot see
the future. The Kelly strategy and the fixed-weight benchmarks are both just
`weight_fn`s, so one engine serves all curves in the README chart.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_backtest.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/kellyfolio/backtest.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_backtest.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (kelly + estimate + data + backtest)

- [ ] **Step 6: Commit**

```bash
git add src/kellyfolio/backtest.py tests/test_backtest.py
git commit -m "feat: walk-forward backtest engine with costs and metrics"
```

---

### Task 6: `report.py` + CLI — today's allocation

**Files:**
- Create: `src/kellyfolio/report.py`, `src/kellyfolio/__main__.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/kellyfolio/report.py`**

```python
"""Today's suggested allocation — the part Tony actually uses each month."""
import pandas as pd

from kellyfolio.estimate import estimate_mu_sigma
from kellyfolio.kelly import kelly_weights


def kelly_strategy(config: dict):
    """Wrap config into a weight_fn usable by both backtest and report."""
    def weight_fn(history: pd.DataFrame):
        try:
            mu, sigma = estimate_mu_sigma(history, config["lookback_months"])
        except ValueError:
            return None  # warm-up: not enough history yet
        return kelly_weights(mu, sigma, config["risk_free_rate"],
                             config["kelly_fraction"])
    return weight_fn


def format_report(weights: pd.Series, portfolio_value: float) -> str:
    rows = dict(weights)
    rows["Cash"] = max(0.0, 1.0 - weights.sum())
    lines = [f"Target allocation for ${portfolio_value:,.0f}:", ""]
    for name, w in rows.items():
        lines.append(f"  {name:<5} {w * 100:5.1f}%   ${w * portfolio_value:,.0f}")
    return "\n".join(lines)
```

- [ ] **Step 4: Implement `src/kellyfolio/__main__.py`**

```python
"""CLI: python -m kellyfolio report --value 10000"""
import argparse

import yaml

from kellyfolio.data import get_prices
from kellyfolio.report import kelly_strategy, format_report


def main():
    parser = argparse.ArgumentParser(prog="kellyfolio")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="print today's suggested allocation")
    report.add_argument("--value", type=float, default=10_000,
                        help="portfolio value in dollars")
    report.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prices = get_prices(config["tickers"], config["start_date"])
    weights = kelly_strategy(config)(prices)
    if weights is None:
        raise SystemExit("Not enough price history for the configured lookback.")
    print(format_report(weights, args.value))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `.venv/bin/pytest tests/test_report.py -v`
Expected: 2 passed

- [ ] **Step 6: Smoke-test the CLI for real (network required)**

Run: `.venv/bin/python -m kellyfolio report --value 10000`
Expected: a printed allocation table for SPY/TLT/GLD + Cash. Sanity-check that weights are between 0 and 1 and sum ≤ 1.

- [ ] **Step 7: Commit**

```bash
git add src/kellyfolio/report.py src/kellyfolio/__main__.py tests/test_report.py
git commit -m "feat: allocation report CLI (python -m kellyfolio report)"
```

---

### Task 7: Results script — the README chart

**Files:**
- Create: `scripts/make_charts.py`
- Create (generated): `assets/backtest.png`, `assets/metrics.md`

- [ ] **Step 1: Implement `scripts/make_charts.py`**

```python
"""Generate the README chart and metrics table from a real backtest.

Compares full / half / quarter Kelly against buy-and-hold SPY and a 60/40
(SPY/TLT) portfolio, all run through the same walk-forward engine.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from kellyfolio.backtest import run_backtest, fixed_weights, cagr, max_drawdown, sharpe
from kellyfolio.data import get_prices
from kellyfolio.report import kelly_strategy

config = yaml.safe_load(open("config.yaml"))
prices = get_prices(config["tickers"], config["start_date"])

strategies = {
    "Full Kelly": kelly_strategy({**config, "kelly_fraction": 1.0}),
    "Half Kelly": kelly_strategy({**config, "kelly_fraction": 0.5}),
    "Quarter Kelly": kelly_strategy({**config, "kelly_fraction": 0.25}),
    "Buy & Hold SPY": fixed_weights(pd.Series({"SPY": 1.0})),
    "60/40 (SPY/TLT)": fixed_weights(pd.Series({"SPY": 0.6, "TLT": 0.4})),
}

cost = config["transaction_cost"]
results = {name: run_backtest(prices, fn, cost) for name, fn in strategies.items()}

Path("assets").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
for name, res in results.items():
    ax.plot(res.equity, label=name)
ax.set_yscale("log")
ax.set_title("Growth of $1 — walk-forward, monthly rebalance, 0.1% costs")
ax.legend()
fig.tight_layout()
fig.savefig("assets/backtest.png", dpi=150)

rows = [
    f"| {name} | {cagr(r.equity):.1%} | {max_drawdown(r.equity):.1%} | {sharpe(r.equity, config['risk_free_rate']):.2f} |"
    for name, r in results.items()
]
table = ["| Strategy | CAGR | Max Drawdown | Sharpe |", "|---|---|---|---|"] + rows
Path("assets/metrics.md").write_text("\n".join(table) + "\n")
print("\n".join(table))
```

- [ ] **Step 2: Run it (network required)**

Run: `.venv/bin/python scripts/make_charts.py`
Expected: prints a metrics table; creates `assets/backtest.png` and `assets/metrics.md`.
Sanity checks: Full Kelly should show the deepest max drawdown; Half/Quarter Kelly drawdowns should be progressively shallower. If the warm-up consumes the first 5 years, curves start ~2011 — that is correct behavior, not a bug.

- [ ] **Step 3: Allow `assets/` in git (it's generated but belongs in the README)**

Verify `.gitignore` does not exclude `assets/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_charts.py assets/
git commit -m "feat: backtest comparison chart and metrics for README"
```

---

### Task 8: Notebook — the capstone story

**Files:**
- Create: `notebooks/kelly_story.ipynb`

- [ ] **Step 1: Create the notebook**

Build `notebooks/kelly_story.ipynb` with this cell outline (markdown cells written for a recruiter skimming, code cells reusing the package — no logic duplicated in the notebook):

1. **MD — Title + hook:** "What is the Kelly Criterion and why ½-Kelly beats full Kelly in real portfolios" + one-paragraph capstone summary.
2. **MD — The math in plain words:** single-asset Kelly = (μ−r)/σ²; what overbetting does to log growth (the growth curve is a parabola — betting 2× Kelly earns *zero* long-run growth).
3. **Code:** load config + prices via `kellyfolio.data.get_prices`.
4. **Code:** today's μ, Σ and full/half/quarter Kelly weights — show how sensitive weights are to μ (recompute with μ ± 2% to demonstrate estimation error; this is the "own the flaws" cell).
5. **Code:** run the five backtests (same code as `scripts/make_charts.py`), plot growth curves and drawdown curves.
6. **MD — Honest limitations:** estimation error, regime changes, costs/taxes, survivorship of the ETF universe.
7. **MD — Conclusion + roadmap:** fractional Kelly as a practical compromise; Phases 2–3.

- [ ] **Step 2: Execute the notebook top to bottom**

Run: `.venv/bin/pip install jupyter && .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/kelly_story.ipynb`
Expected: executes without errors; outputs saved in the notebook so it renders with charts on GitHub.

- [ ] **Step 3: Commit**

```bash
git add notebooks/kelly_story.ipynb
git commit -m "docs: capstone story notebook with live backtests"
```

---

### Task 9: CI + README

**Files:**
- Create: `.github/workflows/ci.yml`, `README.md`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev]"
      - run: pytest -v
```

(Tests use synthetic data and injected downloaders, so CI needs no network access to Yahoo.)

- [ ] **Step 2: Write `README.md`**

Sections, in order:
1. **Title + badges** — CI badge (`https://github.com/<user>/kelly-portfolio-lab/actions/workflows/ci.yml/badge.svg`), Python version.
2. **One-paragraph pitch** — from capstone to working allocator; link to the capstone PDF.
3. **Headline chart** — embed `assets/backtest.png`, then the metrics table from `assets/metrics.md`.
4. **Methodology (the trust section)** — four short bullets: walk-forward with no look-ahead; fractional Kelly vs 60/40 and SPY; transaction costs included; estimation from historical means **with the limitation stated plainly**.
5. **How to run** — `pip install -e .`, `python -m kellyfolio report --value 10000`, `pytest`.
6. **Project structure** — the file tree with one line per module.
7. **Limitations** — honest list (estimation error, no taxes, monthly granularity, ETF universe chosen with hindsight).
8. **Roadmap** — Phase 2 Streamlit dashboard, Phase 3 Alpaca paper trading → real money.

- [ ] **Step 3: Final full test run**

Run: `.venv/bin/pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "docs: README with results, methodology, and CI"
```

---

### Task 10: Publish to GitHub

- [ ] **Step 1: Create the GitHub repo and push** (public — it's a portfolio piece)

Run: `gh repo create kelly-portfolio-lab --public --source . --push`
(If `gh` is not authenticated: `gh auth login` first, or create the repo in the GitHub UI and `git remote add origin … && git push -u origin main`.)

- [ ] **Step 2: Verify CI is green**

Run: `gh run watch` (or check the Actions tab). The README badge must show passing.

- [ ] **Step 3: Confirm the README renders** — chart visible, notebook opens on GitHub with outputs.
