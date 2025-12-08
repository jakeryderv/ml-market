import os
import pandas as pd
import numpy as np
import yfinance as yf


# =============================================================================
# PRICE-BASED FEATURES
# =============================================================================


def add_lagged_returns(df, lags=range(16)):
    """Add lagged daily return features."""
    df = df.copy()
    daily_returns = df["Close"].pct_change(fill_method=None)

    for lag in lags:
        df[f"ret_lag_{lag}"] = daily_returns.shift(lag)

    return df


def add_trend_indicators(df):
    """Add trend-based indicators (trimmed to 10 features)."""
    df = df.copy()
    close = df["Close"]

    # Price/SMA ratios only (most important trend features)
    for w in [20, 50, 200]:
        sma = close.rolling(w, min_periods=w).mean()
        df[f"trend_close_sma_{w}_ratio"] = close / sma

    # MACD
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    df["trend_macd"] = ema12 - ema26
    df["trend_macd_signal"] = (
        df["trend_macd"].ewm(span=9, min_periods=9, adjust=False).mean()
    )
    df["trend_macd_hist"] = df["trend_macd"] - df["trend_macd_signal"]

    # ADX (Average Directional Index)
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = low.shift() - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    alpha = 1 / 14
    atr14 = tr.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / atr14)
    minus_di = 100 * (minus_dm_smooth / atr14)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    df["trend_adx"] = dx.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    df["trend_plus_di"] = plus_di
    df["trend_minus_di"] = minus_di

    return df


def add_momentum_indicators(df, rsi_period=14, stoch_period=14):
    """Add momentum-based indicators."""
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    alpha = 1 / rsi_period
    avg_gain = gain.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["mom_rsi"] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    lowest_low = low.rolling(stoch_period, min_periods=stoch_period).min()
    highest_high = high.rolling(stoch_period, min_periods=stoch_period).max()
    df["mom_stoch_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low)
    df["mom_stoch_d"] = df["mom_stoch_k"].rolling(3, min_periods=3).mean()

    # Williams %R
    df["mom_williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low)

    # Rate of Change
    for w in [5, 10, 20]:
        df[f"mom_roc_{w}"] = (close / close.shift(w) - 1) * 100

    # CCI
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20, min_periods=20).mean()
    mad = tp.rolling(20, min_periods=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["mom_cci"] = (tp - sma_tp) / (0.015 * mad)

    return df


def add_volatility_indicators(df, bb_period=20, atr_period=14):
    """Add volatility-based indicators."""
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    # Bollinger Bands
    sma = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std()
    df["vol_bb_upper"] = sma + 2 * std
    df["vol_bb_lower"] = sma - 2 * std
    df["vol_bb_pct"] = (close - df["vol_bb_lower"]) / (
        df["vol_bb_upper"] - df["vol_bb_lower"]
    )
    df["vol_bb_width"] = (df["vol_bb_upper"] - df["vol_bb_lower"]) / sma

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    df["vol_atr"] = tr.rolling(atr_period, min_periods=atr_period).mean()
    df["vol_atr_pct"] = df["vol_atr"] / close

    # Historical Volatility
    returns = close.pct_change(fill_method=None)
    for w in [5, 10, 20]:
        df[f"vol_hist_{w}"] = returns.rolling(w, min_periods=w).std() * np.sqrt(252)

    # Garman-Klass Volatility
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / df["Open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df["vol_garman_klass"] = np.sqrt(gk.rolling(20, min_periods=20).mean() * 252)

    # Parkinson Volatility
    df["vol_parkinson"] = np.sqrt(
        (1 / (4 * np.log(2))) * log_hl.rolling(20, min_periods=20).mean() * 252
    )

    return df


def add_volume_indicators(df):
    """Add volume-based indicators."""
    df = df.copy()
    close, volume = df["Close"], df["Volume"]
    high, low = df["High"], df["Low"]

    # Volume Moving Averages
    for w in [5, 10, 20]:
        df[f"vlm_sma_{w}"] = volume.rolling(w, min_periods=w).mean()
        df[f"vlm_ratio_{w}"] = volume / df[f"vlm_sma_{w}"]

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["vlm_obv"] = obv
    df["vlm_obv_sma"] = obv.rolling(20, min_periods=20).mean()

    # Accumulation/Distribution
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)
    df["vlm_ad"] = (mfm * volume).cumsum()

    # MFI
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(), 0).rolling(14, min_periods=14).sum()
    neg_mf = mf.where(tp < tp.shift(), 0).rolling(14, min_periods=14).sum()
    df["vlm_mfi"] = 100 - (100 / (1 + pos_mf / neg_mf))

    # VWAP
    df["vlm_vwap_20"] = (tp * volume).rolling(
        20, min_periods=20
    ).sum() / volume.rolling(20, min_periods=20).sum()

    return df


def add_pattern_indicators(df):
    """Add price pattern indicators (trimmed to 12 features)."""
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    # Distance from N-day high/low (key windows only)
    for w in [20, 50]:
        highest = high.rolling(w, min_periods=w).max()
        lowest = low.rolling(w, min_periods=w).min()
        df[f"pat_dist_high_{w}"] = (close - highest) / highest
        df[f"pat_dist_low_{w}"] = (close - lowest) / lowest
        df[f"pat_range_pos_{w}"] = (close - lowest) / (highest - lowest)

    # Candle patterns
    body = close - df["Open"]
    full_range = high - low
    df["pat_body_pct"] = body / full_range
    df["pat_lower_shadow"] = (
        pd.concat([close, df["Open"]], axis=1).min(axis=1) - low
    ) / full_range

    # Gap indicators
    df["pat_gap"] = (df["Open"] - close.shift()) / close.shift()

    return df


# =============================================================================
# VIX & CBOE FEATURES
# =============================================================================


def add_vix_features(df):
    """Add VIX and volatility regime features."""
    df = df.copy()

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")

    try:
        vix = yf.download(
            "^VIX", start=start, end=end, progress=False, auto_adjust=True
        )
        vix = vix.reindex(df.index)
        vix_close = (
            vix["Close"].squeeze()
            if isinstance(vix["Close"], pd.DataFrame)
            else vix["Close"]
        )

        df["vix_close"] = vix_close
        df["vix_change"] = vix_close.pct_change(fill_method=None)
        df["vix_sma_10"] = vix_close.rolling(10, min_periods=10).mean()
        df["vix_sma_20"] = vix_close.rolling(20, min_periods=20).mean()
        df["vix_sma_10_ratio"] = vix_close / df["vix_sma_10"]
        df["vix_sma_20_ratio"] = vix_close / df["vix_sma_20"]
        df["vix_zscore"] = (
            vix_close - vix_close.rolling(60).mean()
        ) / vix_close.rolling(60).std()
        df["vix_high_regime"] = (vix_close > 20).astype(int)
        df["vix_low_regime"] = (vix_close < 15).astype(int)
        df["vix_spike"] = (df["vix_change"] > 0.1).astype(int)

    except Exception as e:
        print(f"  Warning: Could not download VIX: {e}")
        return df

    # VIX Term Structure
    try:
        vix3m = yf.download(
            "^VIX3M", start=start, end=end, progress=False, auto_adjust=True
        )
        vix3m = vix3m.reindex(df.index)
        vix3m_close = (
            vix3m["Close"].squeeze()
            if isinstance(vix3m["Close"], pd.DataFrame)
            else vix3m["Close"]
        )

        df["vix_term_slope"] = vix3m_close / vix_close - 1
        df["vix_contango"] = (df["vix_term_slope"] > 0.05).astype(int)
        df["vix_backwardation"] = (df["vix_term_slope"] < -0.05).astype(int)

    except Exception as e:
        print(f"  Warning: Could not download VIX3M: {e}")

    # VIX vs Realized Volatility
    returns = df["Close"].pct_change(fill_method=None)
    realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100
    df["vix_rv_spread"] = vix_close - realized_vol

    return df


def add_cboe_features(df):
    """Add CBOE volatility indices features."""
    df = df.copy()

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")

    # VVIX
    try:
        vvix = yf.download(
            "^VVIX", start=start, end=end, progress=False, auto_adjust=True
        )
        vvix = vvix.reindex(df.index)
        vvix_close = (
            vvix["Close"].squeeze()
            if isinstance(vvix["Close"], pd.DataFrame)
            else vvix["Close"]
        )

        df["cboe_vvix"] = vvix_close
        df["cboe_vvix_change"] = vvix_close.pct_change(fill_method=None)
        df["cboe_vvix_zscore"] = (
            vvix_close - vvix_close.rolling(60, min_periods=60).mean()
        ) / vvix_close.rolling(60, min_periods=60).std()

        if "vix_close" in df.columns:
            df["cboe_vvix_vix_ratio"] = vvix_close / df["vix_close"]

    except Exception as e:
        print(f"  Warning: Could not download VVIX: {e}")

    # SKEW
    try:
        skew = yf.download(
            "^SKEW", start=start, end=end, progress=False, auto_adjust=True
        )
        skew = skew.reindex(df.index)
        skew_close = (
            skew["Close"].squeeze()
            if isinstance(skew["Close"], pd.DataFrame)
            else skew["Close"]
        )

        df["cboe_skew"] = skew_close
        df["cboe_skew_change"] = skew_close.pct_change(fill_method=None)
        df["cboe_skew_zscore"] = (
            skew_close - skew_close.rolling(60, min_periods=60).mean()
        ) / skew_close.rolling(60, min_periods=60).std()

    except Exception as e:
        print(f"  Warning: Could not download SKEW: {e}")

    # VIX9D
    try:
        vix9d = yf.download(
            "^VIX9D", start=start, end=end, progress=False, auto_adjust=True
        )
        vix9d = vix9d.reindex(df.index)
        vix9d_close = (
            vix9d["Close"].squeeze()
            if isinstance(vix9d["Close"], pd.DataFrame)
            else vix9d["Close"]
        )

        df["cboe_vix9d"] = vix9d_close

        if "vix_close" in df.columns:
            df["cboe_vix_9d_30d_spread"] = vix9d_close / df["vix_close"] - 1

    except Exception as e:
        print(f"  Warning: Could not download VIX9D: {e}")

    return df


# =============================================================================
# FIXED INCOME & CREDIT
# =============================================================================


def add_fixed_income_features(df):
    """Add fixed income and credit spread features (trimmed to 8 features)."""
    df = df.copy()

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")

    # Only need HYG and TLT for spreads
    tickers = {"hyg": "HYG", "tlt": "TLT"}

    bond_data = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            data = data.reindex(df.index)
            close = (
                data["Close"].squeeze()
                if isinstance(data["Close"], pd.DataFrame)
                else data["Close"]
            )
            bond_data[name] = close

            df[f"fi_{name}_ret"] = close.pct_change(fill_method=None)
            df[f"fi_{name}_ret_5d"] = close.pct_change(5, fill_method=None)

        except Exception as e:
            print(f"  Warning: Could not download {ticker}: {e}")

    # Credit spread: HYG vs TLT (key features)
    if "hyg" in bond_data and "tlt" in bond_data:
        df["fi_credit_spread"] = bond_data["hyg"] / bond_data["tlt"]
        df["fi_credit_spread_chg"] = df["fi_credit_spread"].pct_change(fill_method=None)
        df["fi_credit_spread_zscore"] = (
            df["fi_credit_spread"] - df["fi_credit_spread"].rolling(60).mean()
        ) / df["fi_credit_spread"].rolling(60).std()
        df["fi_credit_spread_chg_5d"] = df["fi_credit_spread"].pct_change(
            5, fill_method=None
        )

    return df


# =============================================================================
# CROSS-ASSET FEATURES
# =============================================================================


def add_cross_asset_features(df):
    """Add cross-asset features (trimmed to 8 features)."""
    df = df.copy()

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    spy_ret = df["Close"].pct_change(fill_method=None)

    assets = {"gld": "GLD", "eem": "EEM"}

    asset_data = {}
    for name, ticker in assets.items():
        try:
            data = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            data = data.reindex(df.index)
            close = (
                data["Close"].squeeze()
                if isinstance(data["Close"], pd.DataFrame)
                else data["Close"]
            )
            asset_data[name] = close

            df[f"xa_{name}_ret_5d"] = close.pct_change(5, fill_method=None)
            df[f"xa_{name}_sma_20_ratio"] = close / close.rolling(20).mean()
            df[f"xa_{name}_spy_corr"] = spy_ret.rolling(20).corr(
                close.pct_change(fill_method=None)
            )

        except Exception as e:
            print(f"  Warning: Could not download {ticker}: {e}")

    # Risk on/off (EEM vs GLD)
    if "eem" in asset_data and "gld" in asset_data:
        df["xa_risk_on_off"] = df["xa_eem_ret_5d"] - df["xa_gld_ret_5d"]
        df["xa_eem_gld_ratio"] = asset_data["eem"] / asset_data["gld"]

    return df


# =============================================================================
# SECTOR FEATURES
# =============================================================================


def add_sector_features(df):
    """Add sector dispersion and breadth features."""
    df = df.copy()

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")

    sectors = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLP": "Staples",
        "XLY": "Discretionary",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "RealEstate",
    }

    sector_returns = pd.DataFrame(index=df.index)

    for ticker, name in sectors.items():
        try:
            data = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            data = data.reindex(df.index)
            close = (
                data["Close"].squeeze()
                if isinstance(data["Close"], pd.DataFrame)
                else data["Close"]
            )
            sector_returns[name] = close.pct_change(fill_method=None)
        except Exception as e:
            print(f"  Warning: Could not download {ticker}: {e}")

    if len(sector_returns.columns) < 5:
        print("  Warning: Not enough sectors downloaded")
        return df

    # Cross-sectional dispersion
    df["sect_dispersion_1d"] = sector_returns.std(axis=1)
    df["sect_dispersion_5d"] = df["sect_dispersion_1d"].rolling(5).mean()

    # Breadth
    df["sect_breadth"] = (sector_returns > 0).sum(axis=1) / len(sector_returns.columns)
    df["sect_breadth_5d"] = df["sect_breadth"].rolling(5).mean()

    # Defensive vs Cyclical
    defensive = ["Staples", "Utilities", "Healthcare"]
    cyclical = ["Technology", "Discretionary", "Financials", "Industrials"]

    def_cols = [c for c in defensive if c in sector_returns.columns]
    cyc_cols = [c for c in cyclical if c in sector_returns.columns]

    if def_cols and cyc_cols:
        df["sect_rotation_def_cyc"] = sector_returns[def_cols].mean(
            axis=1
        ) - sector_returns[cyc_cols].mean(axis=1)

    return df


# =============================================================================
# TARGETS
# =============================================================================


def compute_targets(df):
    """Compute 1d/5d returns, direction, and volatility targets."""
    df = df.copy()

    returns = df["Close"].pct_change(fill_method=None)

    # Forward returns
    df["target_ret_1d"] = returns.shift(-1)
    df["target_ret_5d"] = df["Close"].shift(-5) / df["Close"] - 1

    # Direction
    df["target_dir_1d"] = (df["target_ret_1d"] > 0).astype(int)
    df["target_dir_5d"] = (df["target_ret_5d"] > 0).astype(int)

    # Volatility
    df["target_vol_1d"] = returns.abs().shift(-1)
    df["target_vol_5d"] = returns.rolling(5).std().shift(-5)

    return df


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================


def compute_all(df, ticker="SPY"):
    """Compute all features and targets."""
    print("Computing features...")

    print("  Adding price-based features...")
    df = add_lagged_returns(df)
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_pattern_indicators(df)

    print("  Adding VIX features...")
    df = add_vix_features(df)

    print("  Adding CBOE features...")
    df = add_cboe_features(df)

    print("  Adding fixed income features...")
    df = add_fixed_income_features(df)

    print("  Adding cross-asset features...")
    df = add_cross_asset_features(df)

    print("  Adding sector features...")
    df = add_sector_features(df)

    print("  Computing targets...")
    df = compute_targets(df)

    print(f"  Total columns: {len(df.columns)}")

    return df


def get_features(df):
    """Get feature column names."""
    exclude_prefixes = ("target_",)
    exclude_cols = {
        "Date",
        "index",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Adj Close",
    }

    features = [
        col
        for col in df.columns
        if col not in exclude_cols and not col.startswith(exclude_prefixes)
    ]

    return features


def get_feature_categories(features):
    """Categorize features by prefix."""
    categories = {
        "Returns": [],
        "Trend": [],
        "Momentum": [],
        "Volatility": [],
        "Volume": [],
        "Pattern": [],
        "VIX": [],
        "CBOE": [],
        "FixedIncome": [],
        "CrossAsset": [],
        "Sector": [],
        "Other": [],
    }

    prefix_map = {
        "ret_": "Returns",
        "trend_": "Trend",
        "mom_": "Momentum",
        "vol_": "Volatility",
        "vlm_": "Volume",
        "pat_": "Pattern",
        "vix_": "VIX",
        "cboe_": "CBOE",
        "fi_": "FixedIncome",
        "xa_": "CrossAsset",
        "sect_": "Sector",
    }

    for feat in features:
        categorized = False
        for prefix, category in prefix_map.items():
            if feat.startswith(prefix):
                categories[category].append(feat)
                categorized = True
                break
        if not categorized:
            categories["Other"].append(feat)

    categories = {k: v for k, v in categories.items() if v}

    return categories


def summarize_features(df):
    """Print feature summary."""
    features = get_features(df)
    categories = get_feature_categories(features)

    print(f"\nFeature Summary ({len(features)} total)")
    print("=" * 50)
    for cat, feats in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:15s}: {len(feats):3d} features")
    print("=" * 50)
