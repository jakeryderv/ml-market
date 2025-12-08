# Predicting Return and Volatility Distributions for Tech Stocks

## Project Writeup

---

## 1. Problem Definition & Dataset

### Problem Statement

This project investigates whether machine learning can predict the **distribution** of short-term stock returns and volatilitynot just point estimates, but calibrated prediction intervals that quantify uncertainty. The key questions addressed are:

1. Can we predict the distribution of future returns?
2. Can we predict the distribution of future volatility?
3. Do predictions generalize to truly unseen data?

### Why This Problem Matters

Traditional finance assumes markets are efficient, making returns unpredictable. However, **volatility clustering**the tendency for volatile periods to persistis a well-documented phenomenon that can be exploited. This project tests whether ML models can:
- Capture volatility clustering better than classical methods (GARCH)
- Produce well-calibrated uncertainty estimates
- Generalize to out-of-sample data

### Dataset

**Source**: Yahoo Finance via the `yfinance` API

**Universe**: QQQ ETF and its top 10 holdingshigh-volatility tech stocks that provide a challenging test case:
- QQQ, NVDA, MSFT, AAPL, AVGO, AMZN, TSLA, META, GOOGL, GOOG, NFLX

**Time Period**: 2016-01-04 to 2025-11-28 (nearly 10 years)

**Dataset Statistics**:
- **Total samples**: 27,412
- **Features**: 109 technical indicators
- **Targets**: 6 (1D/5D returns, volatility, and direction)
- **Tickers**: 11

**Why This Dataset?**
- **Sufficient history**: Nearly 10 years provides enough data for robust walk-forward validation
- **High volatility**: Tech stocks exhibit strong volatility clustering, making them ideal for testing predictability
- **Liquid markets**: QQQ and its holdings are highly liquid, ensuring data quality
- **Real-world relevance**: These are among the most traded securities globally

---

## 2. Exploratory Data Analysis (EDA)

### Target Distributions

The notebook visualizes all six target variables:

| Target | Mean | Std Dev | Key Insight |
|--------|------|---------|-------------|
| 1D Returns | 0.0012 | 0.0254 | Centered near zero, heavy tails |
| 5D Returns | 0.0059 | 0.0523 | Wider distribution, still centered |
| 1D Volatility | 0.0153 | 0.0134 | Right-skewed, always positive |
| 5D Volatility | 0.0151 | 0.0099 | Smoother than 1D |
| 1D Direction | 53.7% up | - | Slight positive bias |
| 5D Direction | 55.6% up | - | Stronger positive bias |

**Key Observations**:
- Returns are approximately normally distributed but with heavier tails (fat tails indicating extreme events)
- Volatility is right-skewed with occasional spikes (heteroscedastic)
- Direction is slightly biased toward positive (reflecting the long-term upward drift of equity markets)

### Feature-Target Correlations

The top features correlated with 1-day volatility are:

| Feature | Correlation |
|---------|-------------|
| vol_atr_pct | 0.404 |
| vol_garman_klass | 0.385 |
| vol_parkinson | 0.383 |
| vol_hist_20 | 0.343 |
| vol_hist_10 | 0.341 |
| vix_close | 0.298 |

**Insights**:
- Recent volatility measures (ATR, Parkinson, Garman-Klass) are strong predictors
- VIX (market fear gauge) provides cross-asset signal
- Volume ratios also contribute to predictions

### Temporal Distribution

Samples are evenly distributed across years (~2,750 per year), ensuring no temporal bias in training.

---

## 3. Data Cleaning & Preprocessing

### Missing Value Handling

1. **Forward fill** for technical indicators that depend on lagged values
2. **Drop rows** where core OHLCV data or targets are missing
3. **Fill remaining NaN with 0** in the feature matrix (conservative approach for rare edge cases)

### Feature Engineering

109 features are computed from raw OHLCV data, organized into categories:
- **Price-based**: Returns, momentum indicators (ROC, RSI)
- **Volatility**: Historical vol (5/10/20 day), Parkinson, Garman-Klass, ATR
- **Volume**: Volume ratios, OBV
- **VIX**: Levels, spreads, term structure
- **Cross-asset**: Sector ETF signals, fixed income spreads
- **Pattern recognition**: Candlestick patterns

### Train/Test Split Strategy

**Walk-forward cross-validation** with expanding windows:

| Split | Train Years | Test Year |
|-------|-------------|-----------|
| 1 | 2016 | 2017 |
| 2 | 2016-2017 | 2018 |
| 3 | 2016-2018 | 2019 |
| ... | ... | ... |
| 8 | 2016-2023 | 2024 |
| **OOS** | **2016-2024** | **2025** |

**Why this approach?**
- Mimics real-world deployment: always train on past, predict future
- No look-ahead bias
- Tests generalization across different market regimes (bull markets, COVID crash, 2022 bear market)
- Final OOS test on 2025 data ensures no parameter tuning leaked information

---

## 4. Modeling (2+ Models from Different Families)

### Model 1: Random Forest (Ensemble, Tree-Based)

**Configuration**:
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)
```

**Justification**:
- Handles non-linear relationships without feature scaling
- Robust to outliers (important for financial data)
- Provides feature importance for interpretability
- Conservative hyperparameters (shallow trees, high min_samples_leaf) prevent overfitting

**Used for**: Point prediction baselines for returns, direction, and volatility

### Model 2: Quantile Random Forest (Ensemble, Distribution Prediction)

**Configuration**:
```python
RandomForestQuantileRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)
```

**Justification**:
- Extends Random Forest to predict full conditional distributions
- Outputs quantiles (5th, 10th, 25th, 50th, 75th, 90th, 95th percentiles)
- Enables calibrated prediction intervals
- Same hyperparameters as RF for fair comparison

**Used for**: Main distribution prediction results

### Model 3: GARCH(1,1) (Classical Time Series, Parametric)

**Configuration**:
```python
arch_model(returns, vol='Garch', p=1, q=1)
```

**Justification**:
- Industry-standard benchmark for volatility forecasting
- Parametric model from a completely different family (econometric vs ML)
- Uses only univariate price history (no cross-asset features)
- Provides baseline to demonstrate ML value-add

**Model Formula**:
$$\sigma^2_t = \omega + \alpha \cdot \epsilon^2_{t-1} + \beta \cdot \sigma^2_{t-1}$$

### Model 4: Random Forest Classifier (Ensemble, Classification)

**Configuration**: Same hyperparameters as regressor

**Used for**: Direction prediction (up/down) baseline

---

## 5. Model Evaluation & Interpretation

### Metrics Selection

| Task | Metric | Justification |
|------|--------|---------------|
| Regression | Correlation | Scale-invariant, interpretable as signal strength |
| Regression | RMSE | Standard error metric |
| Classification | Accuracy | Simple baseline comparison |
| Classification | AUC | Handles class imbalance |
| Distribution | Coverage | Calibration check (90% CI should contain 90% of actuals) |
| Distribution | Calibration Error | Mean absolute deviation from perfect calibration |

### Point Prediction Results (Random Forest)

| Target | Metric | Value | Interpretation |
|--------|--------|-------|----------------|
| ret_1d | Correlation | 0.021 | Essentially unpredictable |
| ret_5d | Correlation | 0.072 | Slightly better, still weak |
| dir_1d | Accuracy | 52.1% | No better than random (baseline: 53.7%) |
| dir_5d | Accuracy | 54.9% | Marginal improvement |
| vol_1d | Correlation | 0.417 | **Strong predictability** |
| vol_5d | Correlation | 0.565 | **Very strong predictability** |

**Key Insight**: Returns are unpredictable (efficient market hypothesis holds), but volatility is highly predictable (volatility clustering is real and exploitable).

### Distribution Prediction Results (Quantile Random Forest)

| Target | Correlation | 90% Coverage | 50% Coverage |
|--------|-------------|--------------|--------------|
| ret_1d | 0.041 | 87.0% | 47.8% |
| ret_5d | 0.060 | 84.4% | 44.0% |
| vol_1d | 0.405 | 87.6% | 48.6% |
| vol_5d | 0.555 | 88.1% | 48.3% |

**Calibration Interpretation**:
- 90% coverage should be 90%, actual is ~87-88% ’ slightly narrow intervals (minor overconfidence)
- 50% coverage should be 50%, actual is ~48% ’ well-calibrated
- Mean Absolute Calibration Error: Volatility = 1.9%, Returns = 2.6%

### Out-of-Sample Validation (2025)

The ultimate testtrain on 2016-2024, predict 2025 (truly unseen data):

| Metric | In-Sample | OOS (2025) | Change |
|--------|-----------|------------|--------|
| Return Correlation | 0.041 | 0.175 | +0.134 |
| Return 90% Coverage | 87.0% | 82.7% | -4.3% |
| Return 50% Coverage | 47.8% | 42.9% | -4.9% |
| Vol Correlation | 0.405 | 0.448 | +0.043 |
| Vol 90% Coverage | 87.6% | 89.4% | +1.8% |
| Vol 50% Coverage | 48.6% | 50.4% | +1.8% |

**Critical Finding**: Volatility predictions **improve** out-of-sample, demonstrating genuine predictive power rather than overfitting.

### GARCH vs QRF Comparison

| Model | Correlation (QQQ, 252 days) |
|-------|----------------------------|
| GARCH(1,1) | 0.147 |
| Quantile RF | 0.220 |
| **ML Advantage** | **+0.073** |

**Why QRF Wins**: GARCH uses only past returns of one asset. QRF leverages cross-asset features (VIX, sector volatility, volume) for better forecasts.

### Statistical Significance

Bootstrap 95% confidence intervals confirm results are statistically significant:
- Volatility Correlation: 0.405 [0.391, 0.420] 
- Return Correlation: 0.041 [0.025, 0.057]  (significant but economically small)

### Feature Importance

Top features for volatility prediction (permutation importance):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | vol_atr_pct | 0.177 |
| 2 | vlm_ratio_20 | 0.038 |
| 3 | vlm_ratio_10 | 0.013 |
| 4 | vlm_ratio_5 | 0.008 |
| 5 | vol_parkinson | 0.006 |

**Interpretation**: Recent volatility (ATR) dominates, followed by volume dynamics. This aligns with financial intuitionvolatility clusters, and volume spikes often precede volatility.

### Error Analysis

Error patterns reveal model limitations:
- Errors scale with actual volatility (heteroscedastic)
- Model underpredicts extreme volatility events (conservative bias)
- MAE increases from 0.003 (low vol) to 0.012 (very high vol)

---

## Summary

This project demonstrates that:

1. **Returns are unpredictable** (correlation ~0), consistent with efficient market theory
2. **Volatility is predictable** (correlation ~0.4) with well-calibrated uncertainty intervals
3. **Results generalize OOS**: 2025 shows equal or better performance than in-sample
4. **ML beats classical methods**: QRF outperforms GARCH by leveraging cross-asset features

### Practical Applications

- **Risk Management**: Use volatility forecasts for dynamic position sizing
- **Options Pricing**: Better implied volatility estimates with uncertainty bounds
- **Alert Systems**: Flag when actual volatility exceeds predicted intervals

### Limitations

- Returns remain fundamentally unpredictable for point estimates
- Model underpredicts extreme events
- Limited to tech stocks; generalization to other sectors untested
