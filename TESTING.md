# Testing Guide

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_features.py

# Run specific test
uv run pytest tests/test_features.py::test_return_features_schema

# Skip slow integration tests (ones that hit external APIs)
uv run pytest -m "not integration"

# Show test coverage (after adding pytest-cov)
uv run pytest --cov=src/ml_market --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures (synthetic data generators)
├── test_data.py          # Data fetching tests
├── test_features.py      # Feature engineering tests
└── test_model.py         # Model training & validation tests
```

## What We Test

### ✅ Data Layer (`test_data.py`)
- Schema validation (correct columns, types)
- Data consistency (high >= low)
- Date range handling
- Multiple ticker support

### ✅ Features Layer (`test_features.py`)
- Feature schema (all expected columns present)
- No data leakage (features don't use future data)
- Value ranges (RSI 0-100, ATR >= 0)
- Calendar features validity
- Target computation (future returns)
- Full pipeline integration

### ✅ Model Layer (`test_model.py`)
- Walk-forward split logic
- Model training (smoke tests)
- Prediction validity (finite values)
- Feature importance availability

## Test Philosophy

**Do Test:**
- Function contracts (inputs → outputs)
- Data quality checks
- Business logic correctness
- No lookahead bias

**Don't Test:**
- Exact numeric values (market data changes)
- Model performance metrics (too variable)
- External API reliability (not our code)

## Adding New Tests

When adding features or models:

1. **Add fixtures** in `conftest.py` for new data types
2. **Test schemas** - verify expected columns exist
3. **Test invariants** - check constraints always hold
4. **Test edge cases** - empty data, single row, etc.
5. **Integration test** - full pipeline runs without errors

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Before deployment

Current status: **21 tests passing**
