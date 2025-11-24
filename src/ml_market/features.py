import numpy as np
import pandas as pd


# ============================================================
# 1. RETURN & LAG FEATURES
# ============================================================
def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ret1d"] = df.groupby("ticker")["close"].pct_change()
    df["ret5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["ret10d"] = df.groupby("ticker")["close"].pct_change(10)
    df["ret20d"] = df.groupby("ticker")["close"].pct_change(20)
    df["logret"] = np.log(df["close"] / df.groupby("ticker")["close"].shift(1))

    for n in [1, 2, 3, 5, 7, 10]:
        df[f"ret_lag_{n}"] = df.groupby("ticker")["ret1d"].shift(n)
        df[f"close_lag_{n}"] = df.groupby("ticker")["close"].shift(n)

    df["ret_mean_5"] = df.groupby("ticker")["ret1d"].transform(lambda x: x.rolling(5).mean())
    df["ret_std_5"] = df.groupby("ticker")["ret1d"].transform(lambda x: x.rolling(5).std())

    df["ret_std_20"] = df.groupby("ticker")["ret1d"].transform(lambda x: x.rolling(20).std())
    df["ret_vol_ratio_5_20"] = df["ret_std_5"] / df["ret_std_20"]

    df["ret_autocorr_5"] = df.groupby("ticker")["ret1d"].transform(
        lambda x: x.rolling(5).apply(lambda s: s.autocorr(1))
    )

    return df


# ============================================================
# 2. TREND FEATURES
# ============================================================
def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("ticker")["close"]

    for n in [5, 10, 20, 50, 200]:
        df[f"sma_{n}"] = g.transform(lambda x: x.rolling(n).mean())

    for n in [10, 20, 50]:
        df[f"ema_{n}"] = g.transform(lambda x: x.ewm(span=n, adjust=False).mean())

    df["sma_5_20_diff"] = df["sma_5"] - df["sma_20"]
    df["sma_10_50_diff"] = df["sma_10"] - df["sma_50"]

    ema12 = g.transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, adjust=False).mean())

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["price_to_sma20"] = df["close"] / df["sma_20"]
    df["price_to_sma50"] = df["close"] / df["sma_50"]

    df["roc_10"] = g.transform(lambda x: x.pct_change(10))
    df["roc_20"] = g.transform(lambda x: x.pct_change(20))

    return df


# ============================================================
# 3. MOMENTUM FEATURES
# ============================================================
def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def compute_rsi(x, period=14):
        delta = x.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = df.groupby("ticker")["close"].transform(lambda x: compute_rsi(x, 14))

    low14 = df.groupby("ticker")["low"].transform(lambda x: x.rolling(14).min())
    high14 = df.groupby("ticker")["high"].transform(lambda x: x.rolling(14).max())

    df["stoch_k"] = (df["close"] - low14) / (high14 - low14) * 100
    df["stoch_d"] = df.groupby("ticker")["stoch_k"].transform(lambda x: x.rolling(3).mean())
    df["williams_r"] = (high14 - df["close"]) / (high14 - low14) * -100

    return df


# ============================================================
# 4. VOLATILITY FEATURES
# ============================================================
def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    prev_close = df.groupby("ticker")["close"].shift(1)
    df["truerange"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)),
    )

    df["atr_14"] = df.groupby("ticker")["truerange"].transform(lambda x: x.rolling(14).mean())

    df["atr_pct"] = df["atr_14"] / df["close"]

    mid = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    std = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).std())

    df["bb_middle"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["boll_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["vol20"] = df.groupby("ticker")["ret1d"].transform(lambda x: x.rolling(20).std())

    return df


# ============================================================
# 5. VOLUME FEATURES
# ============================================================
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["volume_change"] = df.groupby("ticker")["volume"].pct_change()
    df["volume_ma_20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    df["volume_trend_5"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(5).apply(lambda s: np.polyfit(range(len(s)), s, 1)[0], raw=False)
    )

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    df["vwap"] = (df["typical_price"] * df["volume"]).groupby(df["ticker"]).cumsum() / df[
        "volume"
    ].groupby(df["ticker"]).cumsum()

    df["vwap_dev"] = df["close"] / df["vwap"] - 1

    return df


# ============================================================
# 6. CALENDAR FEATURES
# ============================================================


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    df["is_monday"] = (df["day_of_week"] == 0).astype(int)

    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["days_to_month_end"] = (df["date"] + pd.offsets.MonthEnd(0) - df["date"]).dt.days

    return df


# ============================================================
# 7. TARGETS
# ============================================================
def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["target_ret_1d"] = df.groupby("ticker")["close"].pct_change().shift(-1)
    df["target_ret_5d"] = df.groupby("ticker")["close"].pct_change(5).shift(-5)
    df["target_ret_10d"] = df.groupby("ticker")["close"].pct_change(10).shift(-10)

    df["target_up_1d"] = (df["target_ret_1d"] > 0).astype(int)

    thr = 0.001
    df["target_3c_1d"] = np.where(
        df["target_ret_1d"] > thr, 1, np.where(df["target_ret_1d"] < -thr, -1, 0)
    )

    return df


# ============================================================
# A) SECTOR & MACRO MERGE
# ============================================================
def merge_sector_macro(df, sector_df, macro_df):
    df = df.copy()
    df = df.merge(sector_df, on="date", how="left")
    df = df.merge(macro_df, on="date", how="left")
    return df


# ============================================================
# B) SPREAD FEATURES
# ============================================================
def compute_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stock_vs_sector"] = df["ret1d"] - df["xlk_ret_1d"]
    df["stock_vs_spy"] = df["ret1d"] - df["spy_ret_1d"]
    df["qqq_vs_spy"] = df["qqq_ret_1d"] - df["spy_ret_1d"]
    return df


# ============================================================
# C) REGIME LABELING
# ============================================================
def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime_vol"] = pd.qcut(df["vix_vol_20"], q=3, labels=["low", "mid", "high"])
    df["regime_trend"] = pd.qcut(df["spy_mom_10"], q=3, labels=["down", "neutral", "up"])
    df["regime_momentum"] = pd.qcut(df["xlk_mom_10"], q=3, labels=["weak", "neutral", "strong"])
    return df


# ============================================================
# MASTER FUNCTION
# ============================================================
def compute_all_features(df: pd.DataFrame, sector_df=None, macro_df=None):
    df = compute_return_features(df)
    df = compute_trend_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_volume_features(df)
    df = compute_calendar_features(df)
    df = compute_targets(df)

    if sector_df is not None and macro_df is not None:
        df = merge_sector_macro(df, sector_df, macro_df)
        df = compute_spread_features(df)
        df = compute_regime_features(df)

    df = df.dropna().reset_index(drop=True)
    return df
