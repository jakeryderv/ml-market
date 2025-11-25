# Feature Reference

Complete reference of all features computed in the ml-market pipeline. All features are designed to avoid data leakage - they use only current and historical data, never future information (except targets).

## Table of Contents

- [Core Price Features](#core-price-features)
- [Return & Lag Features](#return--lag-features)
- [Trend & Moving Average Features](#trend--moving-average-features)
- [Momentum Indicators](#momentum-indicators)
- [Volatility Features](#volatility-features)
- [Volume Features](#volume-features)
- [Calendar Features](#calendar-features)
- [Sector Features (XLK)](#sector-features-xlk)
- [Market Features (SPY, QQQ, etc.)](#market-features-spy-qqq-etc)
- [Spread Features](#spread-features)
- [Regime Features](#regime-features)
- [Target Variables](#target-variables)

---

## Core Price Features

**Source**: Raw OHLCV data from yfinance

| Feature | Description | Type |
|---------|-------------|------|
| `date` | Trading date | datetime |
| `ticker` | Stock symbol (AAPL, MSFT, etc.) | string |
| `open` | Opening price | float |
| `high` | Highest price of the day | float |
| `low` | Lowest price of the day | float |
| `close` | Closing price (adjusted) | float |
| `volume` | Number of shares traded | int |

---

## Return & Lag Features

**Purpose**: Capture price momentum, mean reversion, and autocorrelation patterns

| Feature | Formula | Description | Lookback |
|---------|---------|-------------|----------|
| `ret1d` | `(close_t / close_t-1) - 1` | 1-day return | 1 day |
| `ret5d` | `(close_t / close_t-5) - 1` | 5-day return | 5 days |
| `ret10d` | `(close_t / close_t-10) - 1` | 10-day return | 10 days |
| `ret20d` | `(close_t / close_t-20) - 1` | 20-day return | 20 days |
| `logret` | `log(close_t / close_t-1)` | Log return (better for volatility) | 1 day |
| `ret_lag_1` | `ret1d` shifted 1 day | Yesterday's return | 2 days |
| `ret_lag_2` | `ret1d` shifted 2 days | 2 days ago return | 3 days |
| `ret_lag_3` | `ret1d` shifted 3 days | 3 days ago return | 4 days |
| `ret_lag_5` | `ret1d` shifted 5 days | 5 days ago return | 6 days |
| `ret_lag_7` | `ret1d` shifted 7 days | 1 week ago return | 8 days |
| `ret_lag_10` | `ret1d` shifted 10 days | 10 days ago return | 11 days |
| `close_lag_1` through `close_lag_10` | Lagged close prices | Historical price levels | 1-11 days |
| `ret_mean_5` | `mean(ret1d, 5-day window)` | 5-day average return | 5 days |
| `ret_std_5` | `std(ret1d, 5-day window)` | Short-term volatility | 5 days |
| `ret_std_20` | `std(ret1d, 20-day window)` | Longer-term volatility | 20 days |
| `ret_vol_ratio_5_20` | `ret_std_5 / ret_std_20` | Volatility regime change | 20 days |
| `ret_autocorr_5` | `autocorr(ret1d, lag=1, 5-day window)` | Short-term momentum/reversal | 5 days |

**No Data Leakage**: All returns use pct_change() or shift() which only look backwards.

---

## Trend & Moving Average Features

**Purpose**: Identify trend direction, strength, and potential reversals

| Feature | Formula | Description | Lookback |
|---------|---------|-------------|----------|
| `sma_5` | `mean(close, 5 days)` | 5-day simple moving average | 5 days |
| `sma_10` | `mean(close, 10 days)` | 10-day SMA | 10 days |
| `sma_20` | `mean(close, 20 days)` | 20-day SMA | 20 days |
| `sma_50` | `mean(close, 50 days)` | 50-day SMA | 50 days |
| `sma_200` | `mean(close, 200 days)` | 200-day SMA (long-term trend) | 200 days |
| `ema_10` | `EMA(close, span=10)` | 10-day exponential moving average | ~10 days |
| `ema_20` | `EMA(close, span=20)` | 20-day EMA | ~20 days |
| `ema_50` | `EMA(close, span=50)` | 50-day EMA | ~50 days |
| `sma_5_20_diff` | `sma_5 - sma_20` | Fast-slow MA crossover signal | 20 days |
| `sma_10_50_diff` | `sma_10 - sma_50` | Medium-term crossover signal | 50 days |
| `macd` | `EMA12 - EMA26` | MACD line (trend momentum) | ~26 days |
| `macd_signal` | `EMA9(macd)` | MACD signal line | ~35 days |
| `macd_hist` | `macd - macd_signal` | MACD histogram (momentum) | ~35 days |
| `price_to_sma20` | `close / sma_20` | Normalized price vs 20-day avg | 20 days |
| `price_to_sma50` | `close / sma_50` | Normalized price vs 50-day avg | 50 days |
| `roc_10` | `(close_t / close_t-10) - 1` | 10-day rate of change | 10 days |
| `roc_20` | `(close_t / close_t-20) - 1` | 20-day rate of change | 20 days |

**No Data Leakage**: All moving averages use rolling windows that only include current and past prices.

---

## Momentum Indicators

**Purpose**: Measure overbought/oversold conditions and momentum strength

| Feature | Formula | Description | Range | Lookback |
|---------|---------|-------------|-------|----------|
| `rsi_14` | `100 - (100 / (1 + RS))` where `RS = avg_gain / avg_loss` | Relative Strength Index | 0-100 | 14 days |
| `stoch_k` | `(close - low14) / (high14 - low14) * 100` | Stochastic %K (fast) | 0-100 | 14 days |
| `stoch_d` | `SMA3(stoch_k)` | Stochastic %D (slow, smoothed) | 0-100 | 17 days |
| `williams_r` | `(high14 - close) / (high14 - low14) * -100` | Williams %R | -100-0 | 14 days |

**Interpretation**:
- **RSI**: >70 = overbought, <30 = oversold
- **Stochastic**: >80 = overbought, <20 = oversold
- **Williams %R**: >-20 = overbought, <-80 = oversold

**No Data Leakage**: All indicators use rolling windows of past data only.

---

## Volatility Features

**Purpose**: Measure price volatility, trading range, and risk

| Feature | Formula | Description | Lookback |
|---------|---------|-------------|----------|
| `truerange` | `max(high - low, abs(high - prev_close), abs(low - prev_close))` | True range (accounts for gaps) | 1 day |
| `atr_14` | `SMA14(truerange)` | Average True Range | 14 days |
| `atr_pct` | `atr_14 / close` | ATR as percentage of price | 14 days |
| `bb_middle` | `SMA20(close)` | Bollinger Band middle line | 20 days |
| `bb_upper` | `bb_middle + 2 * std20` | Bollinger Band upper (2Ã) | 20 days |
| `bb_lower` | `bb_middle - 2 * std20` | Bollinger Band lower (2Ã) | 20 days |
| `boll_width` | `(bb_upper - bb_lower) / bb_middle` | Bollinger Band width (volatility) | 20 days |
| `bb_position` | `(close - bb_lower) / (bb_upper - bb_lower)` | Position within bands (0-1) | 20 days |
| `vol20` | `std(ret1d, 20 days)` | 20-day return volatility | 20 days |

**Interpretation**:
- **ATR**: Higher values = more volatile
- **Bollinger Bands**: Price near upper band = potentially overbought, near lower = oversold
- **bb_position**: >0.8 = near upper band, <0.2 = near lower band

**No Data Leakage**: True range uses previous close (shift(1)), all other features use rolling windows.

---

## Volume Features

**Purpose**: Analyze trading volume patterns and volume-weighted prices

| Feature | Formula | Description | Lookback |
|---------|---------|-------------|----------|
| `volume_change` | `(volume_t / volume_t-1) - 1` | Daily volume change | 1 day |
| `volume_ma_20` | `mean(volume, 20 days)` | 20-day average volume | 20 days |
| `volume_ratio` | `volume / volume_ma_20` | Current vs average volume | 20 days |
| `volume_trend_5` | `slope(volume, 5 days)` | 5-day volume trend (linear fit) | 5 days |
| `typical_price` | `(high + low + close) / 3` | Typical price (volume-weight basis) | Same day |
| `vwap` | Rolling 20-day volume-weighted average price | VWAP (institutional benchmark) | 20 days |
| `vwap_dev` | `(close / vwap) - 1` | Deviation from VWAP | 20 days |

**Interpretation**:
- **volume_ratio**: >1.5 = high volume, <0.5 = low volume
- **vwap_dev**: Positive = trading above VWAP (bullish), negative = below (bearish)

**No Data Leakage**: Volume features use current/past data. VWAP uses 20-day rolling window (fixed from original cumulative version to avoid lookahead).

---

## Calendar Features

**Purpose**: Capture seasonality and day-of-week effects

| Feature | Values | Description |
|---------|--------|-------------|
| `day_of_week` | 0-6 | Monday (0) through Sunday (6) |
| `month` | 1-12 | Calendar month |
| `week_of_year` | 1-53 | ISO week number |
| `is_month_start` | 0/1 | First trading day of month |
| `is_month_end` | 0/1 | Last trading day of month |
| `is_monday` | 0/1 | Monday flag |
| `is_friday` | 0/1 | Friday flag |
| `days_to_month_end` | 0-31 | Days until month end |

**Use Cases**:
- **Monday effect**: Returns often different on Mondays
- **Month-end**: Institutional rebalancing effects
- **January effect**: Small-cap outperformance

**No Data Leakage**: All features extracted from the date itself.

---

## Sector Features (XLK)

**Purpose**: Technology sector (XLK ETF) context for tech stocks

| Feature | Formula | Description | Lookback |
|---------|---------|-------------|----------|
| `xlk_ret_1d` | `(xlk_close_t / xlk_close_t-1) - 1` | Sector daily return | 1 day |
| `xlk_vol_20` | `std(xlk_ret_1d, 20 days)` | Sector volatility | 20 days |
| `xlk_mom_10` | `(xlk_close_t / xlk_close_t-10) - 1` | Sector 10-day momentum | 10 days |

**No Data Leakage**: Computed same way as stock features, using only past data.

---

## Market Features (SPY, QQQ, etc.)

**Purpose**: Broad market context and risk indicators

### Market Indices

For each of **SPY** (S&P 500), **QQQ** (Nasdaq), **TLT** (Treasury bonds), **DXY** (US Dollar), **VIX** (Volatility):

| Feature Pattern | Description | Lookback |
|----------------|-------------|----------|
| `{symbol}_ret_1d` | Daily return | 1 day |
| `{symbol}_vol_20` | 20-day volatility | 20 days |
| `{symbol}_mom_10` | 10-day momentum | 10 days |
| `{symbol}_close` | Closing price/level | Current |

**Specific Symbols**:
- **SPY**: S&P 500 (broad market)
- **QQQ**: Nasdaq 100 (tech-heavy)
- **TLT**: 20+ year Treasury bonds (risk-off)
- **DXY**: US Dollar Index (currency strength)
- **VIX**: CBOE Volatility Index (fear gauge)

**No Data Leakage**: All computed with rolling windows of past data.

---

## Spread Features

**Purpose**: Relative performance vs benchmarks

| Feature | Formula | Description |
|---------|---------|-------------|
| `stock_vs_sector` | `ret1d - xlk_ret_1d` | Stock outperformance vs sector |
| `stock_vs_spy` | `ret1d - spy_ret_1d` | Stock outperformance vs S&P 500 |
| `qqq_vs_spy` | `qqq_ret_1d - spy_ret_1d` | Tech vs broad market strength |

**Interpretation**:
- Positive = outperforming benchmark
- Negative = underperforming benchmark

**No Data Leakage**: Uses same-day returns (all computed from past prices).

---

## Regime Features

**Purpose**: Identify market regimes (volatility, trend, momentum environments)

**  Updated to avoid data leakage**: Uses rolling 252-day (1-year) percentile ranks instead of global quantiles.

| Feature | Calculation | Labels | Description |
|---------|------------|--------|-------------|
| `regime_vol` | Rolling percentile of `vix_vol_20` | low / mid / high | Volatility environment |
| `regime_trend` | Rolling percentile of `spy_mom_10` | down / neutral / up | Market trend |
| `regime_momentum` | Rolling percentile of `xlk_mom_10` | weak / neutral / strong | Sector momentum |

**How it works**:
1. For each date, compute where current value ranks in past 252 days (0-100th percentile)
2. Bin into tertiles: <33rd = low/down/weak, 33-67th = mid/neutral, >67th = high/up/strong

**Example**: If VIX volatility is at 80th percentile of past year, regime_vol = "high"

**No Data Leakage**: Uses only past 252 days to compute percentile rank at each point.

---

## Target Variables

**  These intentionally use future data** - they are what we're trying to predict!

| Feature | Formula | Description | Future Lookback |
|---------|---------|-------------|-----------------|
| `target_ret_1d` | `(close_t+1 / close_t) - 1` | Next day return | 1 day ahead |
| `target_ret_5d` | `(close_t+5 / close_t) - 1` | 5-day forward return | 5 days ahead |
| `target_ret_10d` | `(close_t+10 / close_t) - 1` | 10-day forward return | 10 days ahead |
| `target_up_1d` | `1 if target_ret_1d > 0 else 0` | Next day direction (binary) | 1 day ahead |
| `target_3c_1d` | `1 if ret > 0.1%, -1 if ret < -0.1%, else 0` | 3-class direction (up/flat/down) | 1 day ahead |

**Usage**:
- For regression: Use `target_ret_*`
- For classification: Use `target_up_1d` (binary) or `target_3c_1d` (multiclass)

---

## Data Leakage Prevention Summary

###  Features That Don't Leak

**All features except targets use only**:
- Current values (e.g., `close`, `volume` on day t)
- Past values via `.shift(n)` where n > 0
- Rolling windows via `.rolling(n)` which include current + past n-1 days

###   Changes Made to Prevent Leakage

1. **Regime Features** (Fixed):
   - **Before**: Used `pd.qcut()` on entire dataset ’ knew future distribution
   - **After**: Uses rolling 252-day percentile ranks ’ only past distribution

2. **VWAP** (Improved):
   - **Before**: Cumulative sum from start ’ unusual behavior
   - **After**: 20-day rolling window ’ standard daily VWAP

###  Validation

All features validated with:
- Unit tests for no lookahead bias
- Walk-forward validation in training
- Proper train/test split maintaining temporal order

---

## Feature Count Summary

| Category | Count |
|----------|-------|
| Core OHLCV | 7 |
| Return & Lag | 22 |
| Trend & MA | 17 |
| Momentum | 4 |
| Volatility | 9 |
| Volume | 7 |
| Calendar | 8 |
| Sector (XLK) | 3 |
| Market (5 symbols × 3 features) | 15 |
| Spreads | 3 |
| Regimes | 3 |
| **Total Features** | **98** |
| **Targets** | **5** |

---

## Usage in Model Training

```python
from ml_market.data import fetch_ohlcv, load_sector_data, load_macro_data
from ml_market.features import compute_all_features

# Fetch data
stocks_df = fetch_ohlcv(["AAPL", "MSFT"], start="2020-01-01", end="2024-01-01")
sector_df = load_sector_data(start="2020-01-01", end="2024-01-01")
macro_df = load_macro_data(start="2020-01-01", end="2024-01-01")

# Compute all features
df = compute_all_features(stocks_df, sector_df, macro_df)

# Select features for training (exclude targets and non-numeric)
feature_cols = [c for c in df.columns if c not in
                ["date", "ticker", "target_ret_1d", "target_ret_5d",
                 "target_ret_10d", "target_up_1d", "target_3c_1d"]]

X = df[feature_cols]
y = df["target_ret_5d"]  # or choose your target
```

---

## References

- **Technical Indicators**: Based on standard definitions from technical analysis
- **Regime Detection**: Inspired by quantitative finance regime-switching models
- **Data Leakage Prevention**: Follows guidelines from "Advances in Financial Machine Learning" by Marcos López de Prado

---

*Last Updated: 2025-11-24*
