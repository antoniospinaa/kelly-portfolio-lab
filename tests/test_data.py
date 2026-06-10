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
