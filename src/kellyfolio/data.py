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
    except ValueError:
        raise
    except Exception:
        if cache_path.exists():
            warnings.warn(f"Download failed; using cached prices from {cache_path}. "
                          "Data may be stale.")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        raise
