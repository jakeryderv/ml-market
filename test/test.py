from ml_market.data import fetch_ohlcv, load_sector_data, load_macro_data
from ml_market.features import compute_all_features
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# =========================


# WALK-FORWARD SPLIT MAKER
# =========================
def walk_forward_splits(df_length, test_size=200, step=200):
    splits = []
    start = test_size

    while start + test_size < df_length:
        train_idx = list(range(0, start))
        test_idx = list(range(start, start + test_size))
        splits.append((train_idx, test_idx))
        start += step

    return splits


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_walk_forward(model, X, y, splits, categorical_features=None):
    results = []

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if categorical_features:
            model.fit(X_train, y_train, categorical_feature=categorical_features)
        else:
            model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        corr = np.corrcoef(y_test, preds)[0, 1]
        direction_acc = (np.sign(preds) == np.sign(y_test)).mean()

        print(f"Split {i + 1}: RMSE={rmse:.5f}, Corr={corr:.5f}, DirAcc={direction_acc:.3f}")

        results.append(
            {
                "split": i + 1,
                "rmse": float(rmse),
                "corr": float(corr),
                "direction_acc": float(direction_acc),
            }
        )

    df_res = pd.DataFrame(results)
    print("\n===== SUMMARY =====")
    print(df_res.mean(numeric_only=True))

    return df_res


#############################################################################
# LOAD DATA + COMPUTE FEATURES
#############################################################################

# TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
START = "2015-01-01"
END = "2025-01-01"


stocks_df = fetch_ohlcv(TICKERS, start=START, end=END)
sector_df = load_sector_data(start=START, end=END)
macro_df = load_macro_data(start=START, end=END)

df = compute_all_features(stocks_df, sector_df, macro_df)
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)


# Add ticker code categorical embedding
df["ticker_code"] = df["ticker"].astype("category").cat.codes


#############################################################################
# DEFINE X AND y
#############################################################################

# target_ret_{1,5,10}d , target_up_1d, target_3d_1d
y = df["target_ret_5d"]

X = df[
    [
        "ticker_code",
        # return lags
        "logret",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_5",
        "ret_lag_7",
        "ret_lag_10",
        "close_lag_1",
        "close_lag_2",
        "close_lag_3",
        "close_lag_5",
        "close_lag_7",
        "close_lag_10",
        "ret_mean_5",
        "ret_std_5",
        "ret_std_20",
        "ret_vol_ratio_5_20",
        "ret_autocorr_5",
        # trend/MA/MACD
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_10",
        "ema_20",
        "ema_50",
        "sma_5_20_diff",
        "sma_10_50_diff",
        "macd",
        "macd_signal",
        "macd_hist",
        "price_to_sma20",
        "price_to_sma50",
        "roc_10",
        "roc_20",
        # momentum
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "williams_r",
        # volatility
        "truerange",
        "atr_14",
        "atr_pct",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "boll_width",
        "bb_position",
        "vol20",
        # volume features
        "volume_change",
        "volume_ma_20",
        "volume_ratio",
        "volume_trend_5",
        "typical_price",
        "vwap",
        "vwap_dev",
        # calendar
        "day_of_week",
        "month",
        "week_of_year",
        "is_month_start",
        "is_month_end",
        "is_monday",
        "is_friday",
        "days_to_month_end",
        # macro & sector
        "xlk_ret_1d",
        "xlk_vol_20",
        "xlk_mom_10",
        "spy_ret_1d",
        "spy_vol_20",
        "spy_mom_10",
        "qqq_ret_1d",
        "qqq_vol_20",
        "qqq_mom_10",
        "tlt_ret_1d",
        "tlt_vol_20",
        "tlt_mom_10",
        "dxy_ret_1d",
        "dxy_vol_20",
        "dxy_mom_10",
        "vix_ret_1d",
        "vix_vol_20",
        "vix_mom_10",
        # spreads
        "stock_vs_sector",
        "stock_vs_spy",
        "qqq_vs_spy",
    ]
]


#############################################################################
# MODELS (RF + LightGBM categorical)
#############################################################################

rf_model = RandomForestRegressor(
    n_estimators=400, max_depth=10, max_features="sqrt", n_jobs=-1, random_state=42
)

lgbm_model = LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    force_col_wise=True,
)


def print_feature_importance(model, feature_names, top_n=30):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # descending order
    for i in sorted_idx[:top_n]:
        print(f"{feature_names[i]:30s}  {importances[i]:.6f}")


#############################################################################
# RUN WALK-FORWARD

#############################################################################

splits = walk_forward_splits(len(df), test_size=200, step=200)

print("Num splits:", len(splits))

print("\n===== Random Forest =====")
res_rf = evaluate_walk_forward(rf_model, X, y, splits)
# print(res_rf.mean(numeric_only=True))

# print("\n===== LightGBM (Categorical) =====")
# res_lgbm = evaluate_walk_forward(lgbm_model, X, y, splits, categorical_features=["ticker_code"])
# print(res_lgbm.mean(numeric_only=True))

print("\n=== RF Feature Importance (Top) ===")
print_feature_importance(rf_model, X.columns)

# print("\n=== LGBM Feature Importance (Top) ===")
# print_feature_importance(lgbm_model, X.columns)
