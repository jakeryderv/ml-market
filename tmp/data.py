"""Data fetching and feature engineering for stock market prediction."""

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_ohlcv(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for given tickers.

    Args:
        tickers: List of ticker symbols (e.g., ['AAPL', 'GOOGL'])
        start: Start date 'YYYY-MM-DD' (optional)
        end: End date 'YYYY-MM-DD' (optional)
        period: Period if start/end not provided (default: '1y')

    Returns:
        DataFrame with columns: date, ticker, open, high, low, close, volume
    """
    if not tickers:
        raise ValueError("At least one ticker symbol must be provided")

    all_data = []

    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)

        if start:
            df = ticker_obj.history(start=start, end=end)
        else:
            df = ticker_obj.history(period=period)

        if df.empty:
            continue

        df = df.reset_index()
        df["ticker"] = ticker
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
        all_data.append(df)

    if not all_data:
        raise ValueError(f"No data found for tickers: {tickers}")

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)

    return result


def fetch_sector_data(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch XLK sector ETF data with derived features."""
    ticker = "XLK"
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns.values]

    df = data.reset_index()
    df.rename(columns=str.lower, inplace=True)

    close_col = [c for c in df.columns if "close" in c]
    if not close_col:
        raise ValueError("No close price column found in sector download")
    df.rename(columns={close_col[0]: "close"}, inplace=True)

    df = df[["date", "close"]].sort_values("date")

    # Multi-period returns (1, 2, 3, 5, 10 days)
    for n in [1, 2, 3, 5, 10]:
        df[f"macro_xlk_return_{n}d"] = df["close"].pct_change(n)

    df = df.drop(columns=["close"])

    return df


def fetch_macro_data(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch macro/market context data (SPY, QQQ, TLT, DXY, VIX)."""
    symbols = ["SPY", "QQQ", "TLT", "DX-Y.NYB", "^VIX"]
    rename_map = {"DX-Y.NYB": "DXY", "^VIX": "VIX"}

    frames = []

    for s in symbols:
        raw = yf.download(s, start=start, end=end, auto_adjust=True)

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = ["_".join(col).strip() for col in raw.columns.values]

        df = raw.reset_index()
        df.rename(columns=str.lower, inplace=True)

        close_col = [c for c in df.columns if "close" in c]
        if not close_col:
            raise ValueError(f"No close column found for {s}")

        label = rename_map.get(s, s).lower()
        df.rename(columns={close_col[0]: f"macro_{label}_close"}, inplace=True)
        df = df[["date", f"macro_{label}_close"]].sort_values("date")

        # Multi-period returns (1, 2, 3, 5, 10 days)
        for n in [1, 2, 3, 5, 10]:
            df[f"macro_{label}_return_{n}d"] = df[f"macro_{label}_close"].pct_change(n)

        frames.append(df)

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="date", how="outer")

    return macro.sort_values("date")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    # Multi-period returns (1, 2, 3, 5, 10 days)
    for n in [1, 2, 3, 5, 10]:
        df[f"return_{n}d"] = g["close"].pct_change(n)

    for n in [1, 2, 3, 5, 10]:
        df[f"return_lag_{n}"] = g["return_1d"].shift(n)
        df[f"price_lag_{n}"] = g["close"].shift(n)

    df["return_mean_5"] = g["return_1d"].transform(lambda x: x.rolling(5).mean())
    df["return_mean_20"] = g["return_1d"].transform(lambda x: x.rolling(20).mean())
    df["return_std_5"] = g["return_1d"].transform(lambda x: x.rolling(5).std())
    df["return_std_20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())
    df["return_std_ratio_5_20"] = df["return_std_5"] / df["return_std_20"].replace(0, np.nan)

    df["return_autocorr_5"] = g["return_1d"].transform(
        lambda x: x.rolling(5).apply(lambda s: s.autocorr(1), raw=False)
    )

    return df


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")["close"]

    for n in [5, 10, 20, 50, 200]:
        df[f"trend_sma_{n}"] = g.transform(lambda x, n=n: x.rolling(n).mean())

    for n in [10, 20, 50]:
        df[f"trend_ema_{n}"] = g.transform(lambda x, n=n: x.ewm(span=n, adjust=False).mean())

    df["trend_sma_diff_5_20"] = df["trend_sma_5"] - df["trend_sma_20"]
    df["trend_sma_diff_10_50"] = df["trend_sma_10"] - df["trend_sma_50"]

    ema12 = g.transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["trend_macd"] = ema12 - ema26
    df["trend_macd_signal"] = df.groupby("ticker")["trend_macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["trend_macd_hist"] = df["trend_macd"] - df["trend_macd_signal"]
    df["trend_macd_norm"] = df["trend_macd_hist"] / df["close"]

    df["trend_price_to_sma_20"] = df["close"] / df["trend_sma_20"]
    df["trend_price_to_sma_50"] = df["close"] / df["trend_sma_50"]

    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    def compute_rsi(x: pd.Series, period: int = 14) -> pd.Series:
        delta = x.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["momentum_rsi_14"] = g["close"].transform(lambda x: compute_rsi(x, 14))

    low14 = g["low"].transform(lambda x: x.rolling(14).min())
    high14 = g["high"].transform(lambda x: x.rolling(14).max())
    hl_range = (high14 - low14).replace(0, np.nan)

    df["momentum_stoch_k"] = (df["close"] - low14) / hl_range * 100
    df["momentum_stoch_d"] = g["momentum_stoch_k"].transform(lambda x: x.rolling(3).mean())
    df["momentum_williams_r"] = (high14 - df["close"]) / hl_range * -100

    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    prev_close = g["close"].shift(1)
    df["volatility_true_range"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)),
    )
    df["volatility_atr_14"] = g["volatility_true_range"].transform(
        lambda x: x.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    )
    df["volatility_atr_pct"] = df["volatility_atr_14"] / df["close"]

    mid = g["close"].transform(lambda x: x.rolling(20).mean())
    std = g["close"].transform(lambda x: x.rolling(20).std())
    df["volatility_bb_middle"] = mid
    df["volatility_bb_upper"] = mid + 2 * std
    df["volatility_bb_lower"] = mid - 2 * std
    df["volatility_bb_width"] = (df["volatility_bb_upper"] - df["volatility_bb_lower"]) / df["volatility_bb_middle"].replace(0, np.nan)
    bb_range = (df["volatility_bb_upper"] - df["volatility_bb_lower"]).replace(0, np.nan)
    df["volatility_bb_position"] = (df["close"] - df["volatility_bb_lower"]) / bb_range

    df["volatility_realized_20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())
    df["volatility_range"] = (df["high"] - df["low"]) / df["close"]

    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    # Multi-period volume change (1, 2, 3, 5, 10 days)
    for n in [1, 2, 3, 5, 10]:
        df[f"volume_change_{n}d"] = g["volume"].pct_change(n)

    df["volume_ma_20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio_20"] = df["volume"] / df["volume_ma_20"].replace(0, np.nan)

    def safe_polyfit(s):
        try:
            return np.polyfit(range(len(s)), s, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    df["volume_trend_5"] = g["volume"].transform(
        lambda x: x.rolling(5).apply(safe_polyfit, raw=False)
    )
    df["volume_trend_norm_5"] = df["volume_trend_5"] / df["volume_ma_20"].replace(0, np.nan)

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_pv"] = df["typical_price"] * df["volume"]
    pv_sum = g["_pv"].transform(lambda x: x.rolling(20).sum())
    vol_sum = g["volume"].transform(lambda x: x.rolling(20).sum())
    df["volume_vwap_20"] = pv_sum / vol_sum.replace(0, np.nan)
    df["volume_vwap_dev"] = df["close"] / df["volume_vwap_20"].replace(0, np.nan) - 1
    df = df.drop(columns=["_pv", "typical_price"])

    return df


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")["return_1d"]

    df["stats_skewness_20"] = g.transform(lambda x: x.rolling(20).skew())
    df["stats_kurtosis_20"] = g.transform(lambda x: x.rolling(20).kurt())

    df["stats_sharpe_20"] = df["return_mean_20"] / df["return_std_20"].replace(0, np.nan)
    df["stats_sharpe_5"] = df["return_mean_5"] / df["return_std_5"].replace(0, np.nan)

    return df


def compute_lagged_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    # Consistent lag periods (1, 2, 3, 5, 10 days)
    indicators = ["momentum_rsi_14", "momentum_stoch_k", "volatility_realized_20", "beta_spy_20"]
    lags = [1, 2, 3, 5, 10]

    for ind in indicators:
        if ind in df.columns:
            for n in lags:
                df[f"{ind}_lag_{n}d"] = g[ind].shift(n)

    return df


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")

    # Shift returns within each ticker group to get future returns
    df["target_return_1d"] = g["return_1d"].shift(-1)
    df["target_return_5d"] = g["return_5d"].shift(-5)
    df["target_return_10d"] = g["return_10d"].shift(-10)

    df["target_up_1d"] = (df["target_return_1d"] > 0).astype(int)

    thr = 0.001
    df["target_direction_1d"] = np.where(
        df["target_return_1d"] > thr, 1, np.where(df["target_return_1d"] < -thr, -1, 0)
    )

    return df


def merge_sector_macro(
    df: pd.DataFrame, sector_df: pd.DataFrame, macro_df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(sector_df, on="date", how="left")
    df = df.merge(macro_df, on="date", how="left")
    return df


def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Multi-period spreads (1, 2, 3, 5, 10 days)
    for n in [1, 2, 3, 5, 10]:
        df[f"spread_vs_sector_{n}d"] = df[f"return_{n}d"] - df[f"macro_xlk_return_{n}d"]
        df[f"spread_vs_spy_{n}d"] = df[f"return_{n}d"] - df[f"macro_spy_return_{n}d"]
        df[f"spread_qqq_vs_spy_{n}d"] = df[f"macro_qqq_return_{n}d"] - df[f"macro_spy_return_{n}d"]

    return df


def compute_beta_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling beta to SPY for each ticker."""
    df = df.copy()

    def rolling_beta(group):
        stock_ret = group["return_1d"]
        market_ret = group["macro_spy_return_1d"]
        cov = stock_ret.rolling(window).cov(market_ret)
        var = market_ret.rolling(window).var()
        return cov / var.replace(0, np.nan)

    df["beta_spy_20"] = df.groupby("ticker", group_keys=False).apply(rolling_beta, include_groups=False)
    return df


def compute_indicator_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate of change for key indicators (trees can't compute derivatives)."""
    df = df.copy()
    g = df.groupby("ticker")

    # Multi-period RSI delta (1, 2, 3, 5, 10 days)
    for n in [1, 2, 3, 5, 10]:
        df[f"rsi_delta_{n}d"] = g["momentum_rsi_14"].diff(n)

    return df


# ============================================================
# MASTER FUNCTIONS
# ============================================================
def compute_all_features(
    df: pd.DataFrame,
    sector_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all features for stock prediction.

    Args:
        df: OHLCV dataframe with columns [date, ticker, open, high, low, close, volume]
        sector_df: Sector ETF data (optional)
        macro_df: Macro indicators data (optional)

    Returns:
        DataFrame with all features computed
    """
    df = compute_return_features(df)
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_rolling_stats(df)
    df = compute_targets(df)

    if sector_df is not None and macro_df is not None:
        df = merge_sector_macro(df, sector_df, macro_df)
        df = compute_spread_features(df)
        df = compute_beta_features(df)

    # Lagged indicators (after beta is computed)
    df = compute_lagged_indicators(df)
    # Tree-optimized features
    df = compute_indicator_deltas(df)

    df = df.dropna().reset_index(drop=True)

    return df


def load_dataset(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    include_macro: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: fetch data and compute all features.

    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date (optional)
        include_macro: Whether to include sector/macro features

    Returns:
        DataFrame ready for modeling
    """
    df = fetch_ohlcv(tickers, start=start, end=end)

    if include_macro:
        sector_df = fetch_sector_data(start=start, end=end)
        macro_df = fetch_macro_data(start=start, end=end)
        df = compute_all_features(df, sector_df, macro_df)
    else:
        df = compute_all_features(df)

    return df


def get_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return feature columns grouped by type, excluding non-stationary features."""
    all_cols = df.columns.tolist()

    # Non-stationary or intermediate columns to exclude
    exclude = {
        # Raw data
        "date", "ticker", "open", "high", "low", "close", "volume",
        # Intermediate calculations
        "volatility_true_range", "volatility_bb_middle", "volatility_bb_upper", "volatility_bb_lower",
        # Non-stationary: raw price levels
        "price_lag_1", "price_lag_2", "price_lag_3", "price_lag_5", "price_lag_10",
        "trend_sma_5", "trend_sma_10", "trend_sma_20", "trend_sma_50", "trend_sma_200",
        "trend_ema_10", "trend_ema_20", "trend_ema_50",
        "trend_sma_diff_5_20", "trend_sma_diff_10_50",
        "trend_macd", "trend_macd_signal", "trend_macd_hist",
        "volatility_atr_14",
        "volume_vwap_20",
        "volume_ma_20", "volume_trend_5",
        # Macro close prices (non-stationary)
        "macro_spy_close", "macro_qqq_close", "macro_tlt_close", "macro_dxy_close", "macro_vix_close",
        # Redundant features
        "return_log_1d",  # equivalent to return_1d for small values
        "return_std_20",  # duplicate of volatility_realized_20
        "momentum_williams_r",  # mathematically equivalent to stoch_k (just inverted)
        "momentum_stoch_d",  # smoothed version of stoch_k, keep raw signal
    }

    targets = [c for c in all_cols if c.startswith("target_")]

    features = [c for c in all_cols if c not in exclude and c not in targets]

    return {
        "features": features,
        "targets": targets,
    }


def get_test_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Test version - consistent multi-period feature set (periods: 1, 2, 3, 5, 10)."""
    all_cols = df.columns.tolist()
    periods = [1, 2, 3, 5, 10]

    # Build keep_only set with consistent multi-period structure
    keep_only = set()

    # Returns (multi-period)
    for n in periods:
        keep_only.add(f"return_{n}d")

    # Return lags
    for n in periods:
        keep_only.add(f"return_lag_{n}")

    # Macro returns (6 instruments × 5 periods)
    for macro in ["xlk", "qqq", "spy", "vix", "tlt", "dxy"]:
        for n in periods:
            keep_only.add(f"macro_{macro}_return_{n}d")

    # Spreads (3 spreads × 5 periods)
    for n in periods:
        keep_only.add(f"spread_vs_sector_{n}d")
        keep_only.add(f"spread_vs_spy_{n}d")
        keep_only.add(f"spread_qqq_vs_spy_{n}d")

    # Volume change (5 periods)
    for n in periods:
        keep_only.add(f"volume_change_{n}d")

    # RSI delta (5 periods)
    for n in periods:
        keep_only.add(f"rsi_delta_{n}d")

    # Lagged indicators (4 indicators × 5 lags)
    for ind in ["momentum_rsi_14", "momentum_stoch_k", "volatility_realized_20", "beta_spy_20"]:
        keep_only.add(ind)  # Base indicator
        for n in periods:
            keep_only.add(f"{ind}_lag_{n}d")

    # Stats
    keep_only.add("stats_skewness_20")
    keep_only.add("stats_kurtosis_20")

    targets = [c for c in all_cols if c.startswith("target_")]
    features = [c for c in all_cols if c in keep_only]

    return {
        "features": features,
        "targets": targets,
    }
