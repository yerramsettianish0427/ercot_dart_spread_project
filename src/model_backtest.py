"""
Walk-forward backtest: expanding window, monthly refit.

First 2 months are burn-in. For each later month, train LightGBM on
everything before it and predict that month. An assert checks there's no
overlap between train and test times.

Then a toy directional signal (sign of the predicted spread) with hit rate
and PnL. No transaction costs, no position sizing, not a trading strategy.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
df["year_month"] = df["timestamp"].dt.to_period("M")

FEATURES = ["forecast_error_mw", "net_load_mw", "season_sin", "season_cos", "hour"]

months = sorted(df["year_month"].unique())
BURN_IN_MONTHS = 2

all_preds = []

for i in range(BURN_IN_MONTHS, len(months)):
    test_month = months[i]
    train_mask = df["year_month"] < test_month
    test_mask = df["year_month"] == test_month

    max_train_time = df.loc[train_mask, "timestamp"].max()
    min_test_time = df.loc[test_mask, "timestamp"].min()
    assert max_train_time < min_test_time, (
        f"LEAKAGE: train max {max_train_time} >= test min {min_test_time}"
    )

    X_train = df.loc[train_mask, FEATURES].copy()
    X_train["hour"] = X_train["hour"].astype("category")
    y_train = df.loc[train_mask, "dart_spread"]

    X_test = df.loc[test_mask, FEATURES].copy()
    X_test["hour"] = X_test["hour"].astype("category").cat.set_categories(X_train["hour"].cat.categories)
    y_test = df.loc[test_mask, "dart_spread"]

    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
        min_child_samples=30, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=["hour"])
    y_pred = model.predict(X_test)

    fold_df = df.loc[test_mask, ["timestamp", "dart_spread"]].copy()
    fold_df["predicted"] = y_pred
    fold_df["train_rows"] = train_mask.sum()
    all_preds.append(fold_df)

    print(f"Fold {test_month}: trained on {train_mask.sum()} rows, "
          f"predicted {test_mask.sum()} hours")

backtest = pd.concat(all_preds).reset_index(drop=True)

wf_r2 = r2_score(backtest["dart_spread"], backtest["predicted"])
wf_rmse = np.sqrt(mean_squared_error(backtest["dart_spread"], backtest["predicted"]))

print(f"\n=== Walk-forward out-of-sample (all folds) ===")
print(f"Total OOS hours: {len(backtest)}")
print(f"R-squared: {wf_r2:.4f}")
print(f"RMSE:      ${wf_rmse:.3f}/MWh")

# 1 MW position in the direction of the predicted spread: gain the realized
# spread when right, lose it when wrong. Ignores costs and how DART trading
# actually works (you bid MW into the day-ahead market).
backtest["signal"] = np.sign(backtest["predicted"])
backtest["pnl"] = backtest["signal"] * backtest["dart_spread"]
backtest["correct_direction"] = (
    np.sign(backtest["predicted"]) == np.sign(backtest["dart_spread"])
)

hit_rate = backtest["correct_direction"].mean()
total_pnl = backtest["pnl"].sum()
avg_pnl_per_hour = backtest["pnl"].mean()
sharpe_like = backtest["pnl"].mean() / backtest["pnl"].std() * np.sqrt(24 * 365)

print(f"\n=== Toy directional backtest (no transaction costs) ===")
print(f"Hit rate (correct sign): {hit_rate*100:.1f}%")
print(f"Total PnL (1 MW notional, sum over {len(backtest)} hours): ${total_pnl:,.2f}")
print(f"Avg PnL per hour: ${avg_pnl_per_hour:.3f}")
print(f"Annualized Sharpe-like ratio (hourly, no costs): {sharpe_like:.2f}")

naive_hit_rate_long = (backtest["dart_spread"] > 0).mean()
print(f"\nBaseline, always predict positive spread: {naive_hit_rate_long*100:.1f}% hit rate")
print(f"Baseline, coin flip: 50.0% hit rate")

backtest.to_csv(os.path.join(ROOT, "outputs", "walk_forward_backtest.csv"), index=False)
print("\nSaved walk-forward results to outputs/walk_forward_backtest.csv")
