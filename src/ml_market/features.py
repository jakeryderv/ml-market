import numpy as np
import pandas as pd


# ============================================================
# 1. RETURN & LAG FEATURES
# ============================================================
def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g_close = df.groupby("ticker")["close"]

    # Returns at different horizons (used for lags and other features)
    return_horizons = [1, 5, 10, 20]
    for n in return_horizons:
        df[f"return_{n}d"] = g_close.pct_change(n)

    # Lagged returns - full matrix of lag x horizon combinations (20 features)
    # lag0 = current return at each horizon
    lag_periods = [0, 1, 5, 10, 20]
    for horizon in return_horizons:
        g_ret = df.groupby("ticker")[f"return_{horizon}d"]
        for lag in lag_periods:
            df[f"lag{lag}_return_{horizon}d"] = g_ret.shift(lag)

    # Rolling statistics on returns (computed before dropping return_1d)
    g_return = df.groupby("ticker")["return_1d"]
    df["return_rolling_mean_5d"] = g_return.transform(lambda x: x.rolling(5).mean())
    df["return_rolling_std_5d"] = g_return.transform(lambda x: x.rolling(5).std())
    df["return_rolling_std_20d"] = g_return.transform(lambda x: x.rolling(20).std())
    df["return_volatility_ratio_5d_20d"] = df["return_rolling_std_5d"] / df["return_rolling_std_20d"].replace(0, np.nan)
    df["return_autocorr_5d"] = g_return.transform(
        lambda x: x.rolling(5).apply(lambda s: s.autocorr(1))
    )

    # Drop intermediate return columns (now captured in lag0_return_*)
    df = df.drop(columns=[f"return_{n}d" for n in return_horizons])

    return df


# ============================================================
# 2. TREND FEATURES
# ============================================================
def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")["close"]

    # Simple Moving Averages
    for n in [5, 10, 20, 50, 200]:
        df[f"trend_sma_{n}d"] = g.transform(lambda x, n=n: x.rolling(n).mean())

    # Exponential Moving Averages
    for n in [10, 20, 50]:
        df[f"trend_ema_{n}d"] = g.transform(lambda x, n=n: x.ewm(span=n, adjust=False).mean())

    # SMA crossover signals
    df["trend_sma_diff_5d_20d"] = df["trend_sma_5d"] - df["trend_sma_20d"]
    df["trend_sma_diff_10d_50d"] = df["trend_sma_10d"] - df["trend_sma_50d"]

    # MACD indicator
    ema12 = g.transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, adjust=False).mean())

    df["trend_macd"] = ema12 - ema26
    df["trend_macd_signal"] = df.groupby("ticker")["trend_macd"].transform(lambda x: x.ewm(span=9).mean())
    df["trend_macd_histogram"] = df["trend_macd"] - df["trend_macd_signal"]

    # Price relative to moving averages
    df["trend_price_to_sma_20d"] = df["close"] / df["trend_sma_20d"]
    df["trend_price_to_sma_50d"] = df["close"] / df["trend_sma_50d"]

    # Rate of Change
    df["trend_roc_10d"] = g.transform(lambda x: x.pct_change(10))
    df["trend_roc_20d"] = g.transform(lambda x: x.pct_change(20))

    # ADX (Average Directional Index) - measures trend strength
    prev_high = df.groupby("ticker")["high"].shift(1)
    prev_low = df.groupby("ticker")["low"].shift(1)
    prev_close = df.groupby("ticker")["close"].shift(1)

    # +DM and -DM
    plus_dm = (df["high"] - prev_high).clip(lower=0)
    minus_dm = (prev_low - df["low"]).clip(lower=0)
    # Only keep the larger one
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)

    # True Range for ADX
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)),
    )

    # Smoothed values (14-period Wilder's smoothing) - must be per-ticker
    period = 14
    alpha = 1 / period

    df["_tr"] = tr
    df["_plus_dm"] = plus_dm
    df["_minus_dm"] = minus_dm

    tr_smooth = df.groupby("ticker")["_tr"].transform(
        lambda x: x.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    )
    plus_dm_smooth = df.groupby("ticker")["_plus_dm"].transform(
        lambda x: x.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    )
    minus_dm_smooth = df.groupby("ticker")["_minus_dm"].transform(
        lambda x: x.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    )

    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

    # DX and ADX
    di_sum = plus_di + minus_di
    dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
    df["trend_adx_14d"] = df.groupby("ticker", group_keys=False).apply(
        lambda x: dx.loc[x.index].ewm(alpha=alpha, min_periods=period, adjust=False).mean(),
        include_groups=False,
    )

    df = df.drop(columns=["_tr", "_plus_dm", "_minus_dm"])

    return df


# ============================================================
# 3. MOMENTUM FEATURES
# ============================================================
def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def compute_rsi(x: pd.Series, period: int = 14) -> pd.Series:
        delta = x.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Use Wilder's smoothing (EMA with alpha=1/period) - industry standard
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # Relative Strength Index
    df["momentum_rsi_14d"] = df.groupby("ticker")["close"].transform(lambda x: compute_rsi(x, 14))

    # Stochastic Oscillator
    low14 = df.groupby("ticker")["low"].transform(lambda x: x.rolling(14).min())
    high14 = df.groupby("ticker")["high"].transform(lambda x: x.rolling(14).max())

    # Protect against division by zero when high == low (flat price period)
    hl_range = (high14 - low14).replace(0, np.nan)
    df["momentum_stochastic_k"] = (df["close"] - low14) / hl_range * 100
    df["momentum_stochastic_d"] = df.groupby("ticker")["momentum_stochastic_k"].transform(lambda x: x.rolling(3).mean())
    df["momentum_williams_r"] = (high14 - df["close"]) / hl_range * -100

    return df


# ============================================================
# 4. VOLATILITY FEATURES
# ============================================================
def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # True Range and ATR
    prev_close = df.groupby("ticker")["close"].shift(1)
    df["volatility_true_range"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)),
    )

    df["volatility_atr_14d"] = df.groupby("ticker")["volatility_true_range"].transform(lambda x: x.rolling(14).mean())
    df["volatility_atr_percent"] = df["volatility_atr_14d"] / df["close"]

    # Bollinger Bands
    mid = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    std = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).std())

    df["volatility_bb_middle"] = mid
    df["volatility_bb_upper"] = mid + 2 * std
    df["volatility_bb_lower"] = mid - 2 * std
    # Protect against division by zero (flat price or zero middle)
    df["volatility_bb_width"] = (df["volatility_bb_upper"] - df["volatility_bb_lower"]) / df["volatility_bb_middle"].replace(0, np.nan)
    bb_range = (df["volatility_bb_upper"] - df["volatility_bb_lower"]).replace(0, np.nan)
    df["volatility_bb_position"] = (df["close"] - df["volatility_bb_lower"]) / bb_range

    # Historical volatility
    df["volatility_rolling_20d"] = df.groupby("ticker")["lag0_return_1d"].transform(lambda x: x.rolling(20).std())

    return df


# ============================================================
# 5. VOLUME FEATURES
# ============================================================
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Volume changes and moving average
    df["volume_change_1d"] = df.groupby("ticker")["volume"].pct_change()
    df["volume_sma_20d"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio_to_sma"] = df["volume"] / df["volume_sma_20d"]

    # Volume trend (linear regression slope)
    df["volume_trend_slope_5d"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(5).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=False)
    )

    # Use rolling 20-day VWAP instead of cumulative (more standard for daily bars)
    # VWAP = sum(typical_price * volume) / sum(volume)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["_pv"] = typical_price * df["volume"]
    pv_sum = df.groupby("ticker")["_pv"].transform(lambda x: x.rolling(20).sum())
    vol_sum = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20).sum())
    df["volume_vwap_20d"] = pv_sum / vol_sum.replace(0, np.nan)
    df = df.drop(columns=["_pv"])

    df["volume_vwap_deviation"] = df["close"] / df["volume_vwap_20d"].replace(0, np.nan) - 1

    # OBV (On-Balance Volume) - cumulative volume based on price direction
    df["volume_obv"] = df.groupby("ticker", group_keys=False).apply(
        lambda x: (x["volume"] * np.sign(x["close"].diff())).fillna(0).cumsum(),
        include_groups=False,
    )

    # OBV momentum (rate of change)
    df["volume_obv_sma_20d"] = df.groupby("ticker")["volume_obv"].transform(lambda x: x.rolling(20).mean())
    df["volume_obv_ratio"] = df["volume_obv"] / df["volume_obv_sma_20d"].replace(0, np.nan)

    return df


# ============================================================
# 6. STATISTICAL FEATURES (normalization & distribution shape)
# ============================================================
def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Z-score normalization of key features (rolling 20-day window)
    df["stat_return_zscore_20d"] = df.groupby("ticker")["lag0_return_1d"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std().replace(0, np.nan)
    )
    df["stat_volume_zscore_20d"] = df.groupby("ticker")["volume"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std().replace(0, np.nan)
    )
    df["stat_close_zscore_20d"] = df.groupby("ticker")["close"].transform(
        lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std().replace(0, np.nan)
    )

    # Percentile rank (where does current value sit in rolling distribution)
    df["stat_return_percentile_20d"] = df.groupby("ticker")["lag0_return_1d"].transform(
        lambda x: x.rolling(20).apply(lambda s: (s.iloc[-1] >= s).sum() / len(s) if len(s) > 0 else np.nan, raw=False)
    )
    df["stat_volume_percentile_20d"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20).apply(lambda s: (s.iloc[-1] >= s).sum() / len(s) if len(s) > 0 else np.nan, raw=False)
    )

    # Rolling skewness (asymmetry of return distribution)
    df["stat_return_skew_20d"] = df.groupby("ticker")["lag0_return_1d"].transform(
        lambda x: x.rolling(20).skew()
    )

    # Rolling kurtosis (tail heaviness of return distribution)
    df["stat_return_kurtosis_20d"] = df.groupby("ticker")["lag0_return_1d"].transform(
        lambda x: x.rolling(20).kurt()
    )

    # Min-max normalization (0-1 scale) over rolling window
    def minmax_normalize(x):
        roll_min = x.rolling(20).min()
        roll_max = x.rolling(20).max()
        return (x - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    df["stat_close_minmax_20d"] = df.groupby("ticker")["close"].transform(minmax_normalize)

    return df


# ============================================================
# 7. TARGETS
# ============================================================
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Future return targets (shifted forward)
    df["target_return_1d"] = df.groupby("ticker")["close"].pct_change().shift(-1)
    df["target_return_5d"] = df.groupby("ticker")["close"].pct_change(5).shift(-5)
    df["target_return_10d"] = df.groupby("ticker")["close"].pct_change(10).shift(-10)

    # Binary direction target
    df["target_direction_up_1d"] = (df["target_return_1d"] > 0).astype(int)

    # 3-class direction target (up/neutral/down)
    threshold = 0.001
    df["target_direction_3class_1d"] = np.where(
        df["target_return_1d"] > threshold, 1, np.where(df["target_return_1d"] < -threshold, -1, 0)
    )

    return df


# ============================================================
# A) SECTOR & MACRO MERGE
# ============================================================
def merge_sector_macro(
    df: pd.DataFrame, sector_df: pd.DataFrame, macro_df: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(sector_df, on="date", how="left")
    df = df.merge(macro_df, on="date", how="left")
    return df


# ============================================================
# B) SPREAD FEATURES
# ============================================================
def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Relative performance spreads
    df["spread_stock_vs_sector"] = df["lag0_return_1d"] - df["xlk_ret_1d"]
    df["spread_stock_vs_spy"] = df["lag0_return_1d"] - df["spy_ret_1d"]
    df["spread_qqq_vs_spy"] = df["qqq_ret_1d"] - df["spy_ret_1d"]
    return df


# ============================================================
# MASTER FUNCTION
# ============================================================
def compute_all_features(
    df: pd.DataFrame, sector_df: pd.DataFrame | None = None, macro_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    df = compute_return_features(df)
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_statistical_features(df)
    df = compute_targets(df)

    if sector_df is not None and macro_df is not None:
        df = merge_sector_macro(df, sector_df, macro_df)
        df = compute_spread_features(df)

    df = df.dropna().reset_index(drop=True)
    return df
