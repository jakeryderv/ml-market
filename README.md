## Feature / Description Table

| Column | Description |
|--------|------------|
| **date** | Trading date (aligned across stock, sector, macro). |
| **ticker** | Stock symbol (AAPL, MSFT, etc). |
| **open / high / low / close** | OHLC price values. |
| **volume** | Number of shares traded that day. |

### Return Features
| Column | Description |
|--------|------------|
| ret1d, ret5d, ret10d, ret20d | 1, 5, 10, 20-day percent returns. |
| logret | Log return: log(close_t / close_t-1). |
| ret_lag_n | Return **n days before** (autocorrelation feature). |
| close_lag_n | Price **n days before**. |
| ret_mean_5 | 5-day mean return. |
| ret_std_5, ret_std_20 | Short/long window volatility. |
| ret_vol_ratio_5_20 | Relative volatility ratio. |
| ret_autocorr_5 | Autocorrelation over 5-day window. |

### Trend / Moving Average Features
| Column | Description |
|--------|------------|
| sma_n, ema_n | Simple / exponential moving averages. |
| sma_5_20_diff, sma_10_50_diff | Moving average crossover spreads. |
| macd | EMA12 minus EMA26. |
| macd_signal | 9-day signal line. |
| macd_hist | MACD histogram = macd - signal. |
| price_to_sma20, price_to_sma50 | Normalized price vs moving averages. |
| roc_10, roc_20 | Rate of change over 10 / 20 days. |

### Momentum Indicators
| Column | Description |
|--------|------------|
| rsi_14 | Relative Strength Index (momentum oscillator). |
| stoch_k, stoch_d | Stochastic oscillator fast & smooth lines. |
| williams_r | Williams %R momentum indicator. |

### Volatility Features
| Column | Description |
|--------|------------|
| truerange | True trading range including gaps. |
| atr_14 | Average true range (volatility). |
| atr_pct | ATR normalized by price. |
| bb_middle / bb_upper / bb_lower | Bollinger bands. |
| boll_width | Band width (volatility pressure). |
| bb_position | Position inside bands. |
| vol20 | 20-day return standard deviation. |

### Volume Features
| Column | Description |
|--------|------------|
| volume_change | Daily percentage change in volume. |
| volume_ma_20 | 20-day average volume. |
| volume_ratio | Volume relative to average. |
| volume_trend_5 | 5-day linear regression slope. |
| typical_price | (High + Low + Close) / 3. |
| vwap | Volume-weighted average price. |
| vwap_dev | Price deviation from VWAP. |

### Calendar Features
| Column | Description |
|--------|------------|
| day_of_week, month, week_of_year | Time-based seasonality encodings. |
| is_month_start, is_month_end | Month boundary flags. |
| is_monday, is_friday | Week boundary flags. |
| days_to_month_end | Remaining trading days. |

### Sector (XLK)
| Column | Description |
|--------|------------|
| xlk_ret_1d | Sector return. |
| xlk_vol_20 | Sector volatility. |
| xlk_mom_10 | Sector momentum. |

### Macro / Market (SPY, QQQ, VIX, TNX, DXY)
| Column | Description |
|--------|------------|
| *_ret_1d | Market index/ETF daily return. |
| *_vol_20 | 20-day volatility. |
| *_mom_10 | 10-day momentum. |
| vix_close | Volatility fear index. |

### Spread / Relative Strength
| Column | Description |
|--------|------------|
| stock_vs_sector | Ret vs XLK. |
| stock_vs_spy | Ret vs SPY. |
| qqq_vs_spy | Tech vs broad-market strength. |

### Regime Labels
| Column | Description |
|--------|------------|
| regime_vol | Low / mid / high vol buckets. |
| regime_trend | Up / neutral / down trend buckets. |
| regime_momentum | Weak / neutral / strong momentum. |

### Targets
| Column | Description |
|--------|------------|
| target_ret_1d, target_ret_5d, target_ret_10d | Forward returns (prediction target). |
| target_up_1d | 1 = up next day, 0 = down. |
| target_3c_1d | 3-class: up / flat / down. |
