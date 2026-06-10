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

# Fair comparison: the Kelly strategies only start after their estimation
# warm-up, so trim every curve to the common window and restart each at $1.
# Otherwise the benchmarks absorb 2008 while Kelly never faces it.
common_start = max(r.equity.index[0] for r in results.values())
equity_curves = {
    name: (e := r.equity.loc[common_start:]) / e.iloc[0] for name, r in results.items()
}

Path("assets").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
for name, curve in equity_curves.items():
    ax.plot(curve, label=name)
ax.set_yscale("log")
ax.set_title("Growth of $1 — walk-forward, monthly rebalance, 0.1% costs")
ax.legend()
fig.tight_layout()
fig.savefig("assets/backtest.png", dpi=150)

rows = [
    f"| {name} | {cagr(e):.1%} | {max_drawdown(e):.1%} | {sharpe(e, config['risk_free_rate']):.2f} |"
    for name, e in equity_curves.items()
]
table = ["| Strategy | CAGR | Max Drawdown | Sharpe |", "|---|---|---|---|"] + rows
Path("assets/metrics.md").write_text("\n".join(table) + "\n")
print("\n".join(table))
