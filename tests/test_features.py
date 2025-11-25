"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd

from ml_market.features import (
    compute_all_features,
    compute_calendar_features,
    compute_momentum_features,
    compute_return_features,
    compute_targets,
    compute_trend_features,
    compute_volatility_features,
)


def test_return_features_schema(sample_ohlcv_data):
    """Test that return features produces expected columns."""
    result = compute_return_features(sample_ohlcv_data)

    expected_cols = [
        "ret1d",
        "ret5d",
        "ret10d",
        "ret20d",
        "logret",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_7",
        "ret_lag_10",
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_5",
        "close_lag_7",
        "close_lag_10",
        "ret_mean_5",
        "ret_std_5",
        "ret_std_20",
        "ret_vol_ratio_5_20",
        "ret_autocorr_5",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_return_features_no_leakage(sample_ohlcv_data):
    """Test that return features don't use future data."""
    df = compute_return_features(sample_ohlcv_data)

    # ret_lag_1 should be yesterday's return
    # For a given row i, ret_lag_1[i] should equal ret1d[i-1]
    ticker_data = df[df["ticker"] == "AAPL"].reset_index(drop=True)

    # Skip NaN rows
    valid_idx = ticker_data["ret_lag_1"].notna()
    ret_lag = ticker_data.loc[valid_idx, "ret_lag_1"].values
    ret1d_shifted = ticker_data.loc[valid_idx, "ret1d"].shift(1).values

    # Should be approximately equal (allowing for floating point error)
    assert np.allclose(ret_lag[1:], ret1d_shifted[1:], rtol=1e-10, equal_nan=True)


def test_trend_features_schema(sample_ohlcv_data):
    """Test that trend features produces expected columns."""
    result = compute_trend_features(sample_ohlcv_data)

    expected_cols = [
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_10",
        "ema_20",
        "ema_50",
        "sma_5_20_diff",
        "sma_10_50_diff",
        "macd",
        "macd_signal",
        "macd_hist",
        "price_to_sma20",
        "price_to_sma50",
        "roc_10",
        "roc_20",
    ]

    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_momentum_features_rsi_range(sample_ohlcv_data):
    """Test that RSI values are in valid range [0, 100]."""
    result = compute_momentum_features(sample_ohlcv_data)

    rsi_valid = result["rsi_14"].dropna()
    assert (rsi_valid >= 0).all(), "RSI should be >= 0"
    assert (rsi_valid <= 100).all(), "RSI should be <= 100"


def test_volatility_features_atr_positive(sample_ohlcv_data):
    """Test that ATR (Average True Range) is always positive."""
    # Need to compute returns first (vol20 depends on ret1d)
    df = compute_return_features(sample_ohlcv_data)
    result = compute_volatility_features(df)

    atr_valid = result["atr_14"].dropna()
    assert (atr_valid >= 0).all(), "ATR should be >= 0"


def test_calendar_features_valid_ranges(sample_ohlcv_data):
    """Test that calendar features are in valid ranges."""
    result = compute_calendar_features(sample_ohlcv_data)

    assert (result["day_of_week"] >= 0).all() and (result["day_of_week"] <= 6).all()
    assert (result["month"] >= 1).all() and (result["month"] <= 12).all()
    assert (result["week_of_year"] >= 1).all() and (result["week_of_year"] <= 53).all()
    assert result["is_monday"].isin([0, 1]).all()
    assert result["is_friday"].isin([0, 1]).all()


def test_targets_no_data_leakage(sample_ohlcv_data):
    """Test that target features use future data correctly (shifted negative)."""
    df = sample_ohlcv_data.copy()
    df = compute_return_features(df)  # Need ret1d first
    result = compute_targets(df)

    # target_ret_1d should be tomorrow's return
    # For row i, target_ret_1d[i] should equal the return from close[i] to close[i+1]
    ticker_data = result[result["ticker"] == "AAPL"].reset_index(drop=True)

    # Manually calculate what target_ret_1d should be
    expected_target = ticker_data["close"].pct_change().shift(-1)
    actual_target = ticker_data["target_ret_1d"]

    # Compare (skip NaN)
    valid_idx = ~expected_target.isna() & ~actual_target.isna()
    assert np.allclose(expected_target[valid_idx], actual_target[valid_idx], rtol=1e-10), (
        "target_ret_1d should be shifted -1 period"
    )


def test_compute_all_features_integration(sample_ohlcv_data, sample_sector_data, sample_macro_data):
    """Integration test: full feature pipeline runs without errors."""
    result = compute_all_features(sample_ohlcv_data, sample_sector_data, sample_macro_data)

    # Note: dropna() in compute_all_features removes rows with any NaN
    # With 100 days and rolling windows up to 200, many rows get dropped
    # Just verify the pipeline runs and produces valid structure
    assert len(result) > 0, "Should have at least some valid rows"

    # Should have target columns
    assert "target_ret_5d" in result.columns
    assert "target_up_1d" in result.columns

    # Should have sector/macro features
    assert "xlk_ret_1d" in result.columns
    assert "spy_ret_1d" in result.columns

    # No NaN after dropna (in compute_all_features)
    assert result.isna().sum().sum() == 0, "Should have no NaN values after processing"


def test_feature_computation_reproducible(sample_ohlcv_data):
    """Test that feature computation is deterministic."""
    result1 = compute_return_features(sample_ohlcv_data.copy())
    result2 = compute_return_features(sample_ohlcv_data.copy())

    pd.testing.assert_frame_equal(result1, result2)
