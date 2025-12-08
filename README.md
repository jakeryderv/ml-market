# Predicting Return and Volatility Distributions for Tech Stocks

A machine learning project demonstrating that while short-term stock returns are largely unpredictable, volatility can be forecasted with meaningful accuracy using quantile regression forests.

## Problem Statement

**Context**: Financial markets exhibit varying degrees of predictability across different targets. While the Efficient Market Hypothesis suggests returns are difficult to predict, volatility displays well-documented clustering behavior that may be exploitable.

**Objective**: Predict the *distribution* of short-term (1-day, 5-day) returns and volatility for tech stocks - not just point estimates, but calibrated prediction intervals that quantify uncertainty.

**Scope**:
- Universe: QQQ ETF and its top 10 holdings (NVDA, MSFT, AAPL, AVGO, AMZN, TSLA, META, GOOGL, GOOG, NFLX)
- Time period: 2016-2024 for training/CV, 2025 for out-of-sample validation
- Prediction horizons: 1-day and 5-day forward returns and volatility

## Dataset Description

**Source**: Yahoo Finance via `yfinance` API

**Fields**:
- **Core OHLCV**: Open, High, Low, Close, Volume (daily)
- **Technical indicators** (109 features total):
  - Price-based: Lagged returns (16), trend (10), momentum (9), volatility (12), volume (14), patterns (12)
  - External data: VIX/CBOE indices (16), fixed income/credit spreads (8), cross-asset (8), sector dispersion (4)
- **Targets**: 1D/5D forward returns, direction (binary), and realized volatility

**Size**:
- 27,412 total samples (11 tickers x ~2,500 trading days)
- 24,904 CV samples (2016-2024), 2,508 OOS samples (2025)

**License**: Yahoo Finance data is subject to their Terms of Service. This project is for educational/research purposes only.

## Approach Summary

### EDA Highlights
- Returns exhibit high kurtosis (12.6) indicating fat tails - extreme events more frequent than normal distribution
- Volatility is right-skewed (3.7) with occasional large spikes
- Strong autocorrelation in volatility (clustering) vs. near-zero for returns
- 53.7% of days are up days (slight positive drift)

### Preprocessing
- Forward-fill missing external data features
- Drop rows with missing core OHLCV or targets
- StandardScaler applied to linear models; tree-based models use raw features

### Models
| Model | Type | Use Case |
|-------|------|----------|
| Ridge Regression | Linear baseline | Returns/volatility point prediction |
| Logistic Regression | Linear baseline | Direction classification |
| Random Forest | Non-linear | Point prediction, feature importance |
| **Quantile Random Forest** | **Distribution** | **Main model - prediction intervals** |
| GARCH(1,1) | Time series baseline | Volatility benchmark |

### Rationale
- **Walk-forward CV**: Expanding window (train on 2016, test 2017 ’ train on 2016-2017, test 2018 ’ ...) prevents look-ahead bias
- **Quantile RF over point predictions**: Captures uncertainty; produces calibrated intervals showing model confidence
- **QRF over GARCH**: Leverages cross-asset features (VIX, volume, sector dispersion) that univariate GARCH cannot use

## Key Results

### Distribution Prediction Performance

| Target | CV Correlation | OOS Correlation | 90% Coverage | 50% Coverage |
|--------|----------------|-----------------|--------------|--------------|
| Returns 1D | 0.044 | 0.188 | 87.0% / 83.0% | 47.7% / 42.9% |
| Returns 5D | 0.061 | - | 84.6% | 44.3% |
| **Volatility 1D** | **0.407** | **0.442** | **87.6% / 89.2%** | **48.6% / 50.6%** |
| Volatility 5D | 0.553 | - | 88.0% | 48.2% |

### Model Comparison (Volatility)
- **QRF vs GARCH**: 0.219 vs 0.147 correlation on QQQ (+49% improvement)
- **RF vs Ridge**: 0.42 vs 0.35 correlation (non-linear patterns matter)

### Calibration
- Mean absolute calibration error: 1.8% (volatility), 1.4% (returns)
- Near-perfect diagonal on calibration plots - model "knows what it doesn't know"

### Feature Importance (Top 5 for Volatility)
1. `vol_atr_pct` (ATR as % of price): 0.177
2. `vlm_ratio_20` (volume vs 20-day avg): 0.038
3. `vlm_ratio_10`: 0.013
4. `vlm_ratio_5`: 0.008
5. `vol_parkinson` (Parkinson volatility): 0.006

### Trading Strategy (Appendix)
- Vol-timing strategy: In-sample Sharpe +0.17 vs buy-and-hold; **OOS degrades to -0.10**
- Max drawdown reduction: -9.1% in-sample, -5.1% OOS
- Conclusion: Statistical predictability does not automatically translate to trading alpha

## Reproducibility

### Requirements
- Python >= 3.11 (tested with 3.11, 3.12)
- Dependencies managed via `uv` (see `pyproject.toml`)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ml-market

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or with pip (generate requirements first)
uv pip compile pyproject.toml -o requirements.txt
pip install -r requirements.txt
```

### Dependencies
```
arch>=8.0.0
ipykernel>=7.1.0
jupyter>=1.1.1
matplotlib>=3.10.7
numpy>=2.3.5
pandas>=2.3.3
quantile-forest>=1.4.1
scikit-learn>=1.7.2
seaborn>=0.13.2
yfinance>=0.2.66
```

### Running the Analysis

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook main_notebook.ipynb

# Or run as script (cells execute sequentially)
jupyter nbconvert --to script main_notebook.ipynb
python main_notebook.py
```

**Note**: First run downloads ~10 years of data for 11 tickers plus VIX, sector ETFs, and other external data. This takes 2-5 minutes depending on connection speed.

## Ethics, Risks, and Limitations

### Data Leakage Risks
- **Mitigated**: Walk-forward CV ensures no future information leaks into training
- **Mitigated**: External data (VIX, sectors) aligned by date, not forward-filled across time boundaries
- **Caution**: Some features use same-day data (e.g., volume) which is available in practice but introduces subtle timing assumptions

### Bias and Limitations
- **Survivorship bias**: Only includes current QQQ top holdings; excludes delisted/removed stocks
- **Selection bias**: Tech-heavy universe may not generalize to other sectors
- **Regime dependence**: Trained primarily on 2016-2024 (low rates, tech bull market); may underperform in different regimes
- **OOS period short**: Only ~230 trading days of true OOS (2025 YTD)

### Practical Limitations
- **Returns remain unpredictable**: 0.04 correlation is not actionable for trading
- **Strategy degradation OOS**: Vol-timing strategy underperforms buy-and-hold in 2025
- **Rare confident signals**: Direction strategy produces only ~1 signal/year
- **Transaction costs ignored**: Real-world implementation would reduce any edge

### Ethical Considerations
- **Not financial advice**: This is an educational project demonstrating ML techniques
- **Market impact**: Strategies based on public data face crowding risk
- **Model opacity**: Random forests are less interpretable than linear models; feature importance helps but doesn't fully explain predictions

### Recommendations for Practitioners
1. Use volatility forecasts for risk management, not alpha generation
2. Always validate on true out-of-sample data before deployment
3. Account for transaction costs, slippage, and capacity constraints
4. Monitor for regime changes that may invalidate historical patterns
