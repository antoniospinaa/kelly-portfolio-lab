"""Today's suggested allocation — the part used in real life each month."""
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
