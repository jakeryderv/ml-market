# ml-market

Machine learning models for stock market prediction using technical indicators and macro features.

## Overview

Predicts 5-day forward returns for tech stocks (AAPL, MSFT, AMZN, GOOGL, META) using:
- 100+ engineered features (technical indicators, momentum, volatility, volume)
- Sector (XLK) and macro data (SPY, QQQ, VIX, TLT, DXY)
- Walk-forward validation (200-day test windows)
- Random Forest regression

## Setup

```bash
uv sync
uv run python scripts/test.py
```

## Project Structure

- `src/ml_market/` - Core modules (data fetching, feature engineering)
- `scripts/` - Training and evaluation scripts
- `features.md` - Feature descriptions
