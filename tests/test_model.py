"""Tests for model training and evaluation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def test_walk_forward_splits():
    """Test walk-forward split generation."""
    from scripts.test import walk_forward_splits

    # 1000 samples, 200 test, 200 step
    splits = walk_forward_splits(1000, test_size=200, step=200)

    # Check we get expected number of splits
    # Start at 200, then 400, 600
    # (800 would need indices up to 999, but 800+200=1000 exceeds bounds)
    # So we get splits at: 200, 400, 600 = 3 splits
    assert len(splits) >= 3, "Should have at least 3 splits"

    # Check no overlap between train/test
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))

    # Check test always comes after train
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)


def test_model_can_train(sample_ohlcv_data, sample_sector_data, sample_macro_data):
    """Smoke test: model can train without errors."""
    from ml_market.features import compute_all_features

    df = compute_all_features(sample_ohlcv_data, sample_sector_data, sample_macro_data)

    # Simple feature set
    features = ["logret", "ret_lag_1", "sma_10", "rsi_14", "volume_ratio"]
    X = df[features].dropna()
    y = df.loc[X.index, "target_ret_5d"]

    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Check predictions
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_predictions_are_finite(sample_ohlcv_data, sample_sector_data, sample_macro_data):
    """Test that model predictions are finite (not inf or nan)."""
    from ml_market.features import compute_all_features

    df = compute_all_features(sample_ohlcv_data, sample_sector_data, sample_macro_data)

    features = ["logret", "ret_lag_1", "sma_10", "rsi_14"]
    X = df[features].dropna()
    y = df.loc[X.index, "target_ret_5d"]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)

    assert np.isfinite(preds).all(), "All predictions should be finite"


def test_feature_importance_exists(sample_ohlcv_data, sample_sector_data, sample_macro_data):
    """Test that trained model has feature importance."""
    from ml_market.features import compute_all_features

    df = compute_all_features(sample_ohlcv_data, sample_sector_data, sample_macro_data)

    features = ["logret", "ret_lag_1", "sma_10"]
    X = df[features].dropna()
    y = df.loc[X.index, "target_ret_5d"]

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_

    assert len(importances) == len(features)
    assert np.isfinite(importances).all()
    assert (importances >= 0).all()
    assert np.isclose(importances.sum(), 1.0)  # Should sum to 1
