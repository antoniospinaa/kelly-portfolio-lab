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
