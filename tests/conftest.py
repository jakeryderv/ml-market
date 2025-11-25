"""Pytest fixtures for ml-market tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Create synthetic OHLCV data for testing."""
    # Use more days to avoid dropna() removing everything
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    tickers = ["AAPL", "MSFT"]

    data = []
    np.random.seed(42)  # For reproducibility
    for ticker in tickers:
        # Create realistic price data
        base_price = 150.0 if ticker == "AAPL" else 300.0
        prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)

        for i, date in enumerate(dates):
            price = prices[i]
            data.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price * 0.99,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price,
                    "volume": np.random.randint(1000000, 10000000),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_sector_data():
    """Create synthetic sector (XLK) data."""
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    np.random.seed(43)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(len(dates)))

    df = pd.DataFrame({"date": dates, "close": prices})
    df["xlk_ret_1d"] = df["close"].pct_change()
    df["xlk_vol_20"] = df["xlk_ret_1d"].rolling(20).std()
    df["xlk_mom_10"] = df["close"].pct_change(10)

    return df


@pytest.fixture
def sample_macro_data():
    """Create synthetic macro data (SPY, QQQ, etc.)."""
    dates = pd.date_range("2023-01-01", periods=300, freq="D")

    df = pd.DataFrame({"date": dates})

    # Generate data for each symbol
    np.random.seed(44)
    for symbol in ["spy", "qqq", "tlt", "dxy", "vix"]:
        base = 100.0 if symbol != "vix" else 15.0
        prices = base + np.cumsum(np.random.randn(len(dates)) * 0.5)

        df[f"{symbol}_close"] = prices
        df[f"{symbol}_ret_1d"] = pd.Series(prices).pct_change()
        df[f"{symbol}_vol_20"] = df[f"{symbol}_ret_1d"].rolling(20).std()
        df[f"{symbol}_mom_10"] = pd.Series(prices).pct_change(10)

    return df
