"""Tests for data fetching functions."""

import pandas as pd
import pytest

from ml_market.data import fetch_ohlcv, load_macro_data, load_sector_data


def test_fetch_ohlcv_schema():
    """Test that fetch_ohlcv returns correct schema (integration test with real API)."""
    # This hits real API - mark as slow or skip in CI
    df = fetch_ohlcv(["AAPL"], period="5d")

    expected_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["ticker"].dtype == object
    assert pd.api.types.is_numeric_dtype(df["close"])


def test_fetch_ohlcv_multiple_tickers():
    """Test fetching multiple tickers."""
    df = fetch_ohlcv(["AAPL", "MSFT"], period="5d")

    assert "AAPL" in df["ticker"].values
    assert "MSFT" in df["ticker"].values
    assert len(df) > 0


def test_fetch_ohlcv_date_range():
    """Test fetching with explicit date range."""
    df = fetch_ohlcv(["AAPL"], start="2024-01-01", end="2024-01-31")

    assert df["date"].min() >= pd.Timestamp("2024-01-01")
    assert df["date"].max() <= pd.Timestamp("2024-01-31")


def test_fetch_ohlcv_empty_ticker_list():
    """Test that empty ticker list raises error."""
    with pytest.raises(ValueError, match="At least one ticker"):
        fetch_ohlcv([])


def test_fetch_ohlcv_high_low_consistency():
    """Test that high >= low for all rows."""
    df = fetch_ohlcv(["AAPL"], period="5d")

    assert (df["high"] >= df["low"]).all(), "High should be >= Low"


def test_fetch_ohlcv_sorted_by_date():
    """Test that data is sorted by date and ticker."""
    df = fetch_ohlcv(["AAPL", "MSFT"], period="5d")

    # Check if sorted
    assert (
        df["date"].is_monotonic_increasing
        or df.groupby("ticker")["date"].apply(lambda x: x.is_monotonic_increasing).all()
    )


@pytest.mark.integration
def test_load_sector_data_schema():
    """Test sector data loading (slow - hits API)."""
    df = load_sector_data(start="2024-01-01", end="2024-01-31")

    expected_cols = ["date", "close", "xlk_ret_1d", "xlk_vol_20", "xlk_mom_10"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.integration
def test_load_macro_data_schema():
    """Test macro data loading (slow - hits API)."""
    df = load_macro_data(start="2024-01-01", end="2024-01-31")

    # Should have columns for all symbols
    for symbol in ["spy", "qqq", "tlt", "dxy", "vix"]:
        assert f"{symbol}_ret_1d" in df.columns
        assert f"{symbol}_vol_20" in df.columns
        assert f"{symbol}_mom_10" in df.columns
