"""
Walk-forward backtest — the only evaluation in this project that should be
trusted for a "does this work" claim. Everything in model_ols.py and
model_lgbm.py used a single 80/20 split, which is a reasonable first check
but is one lucky (or unlucky) split away from being misleading.

Method: expanding window, monthly refit.
  - Start with the first 2 months as a minimum training set (burn-in).
  - For each subsequent month: train LightGBM on everything before it,
    predict that month, record predictions.
  - Never let the model see the future at any point — this is the single
    most important property of the whole backtest and is checked
    explicitly below.

Then: build the simplest possible trading signal (direction of predicted
spread) and report PnL, hit rate, and the stated assumptions/limitations
of that backtest (no transaction costs, no position sizing, no capacity
constraints — explicitly a toy signal, not a trading strategy).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

DATA_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
df["year_month"] = df["timestamp"].dt.to_period("M")

FEATURES = ["forecast_error_mw", "net_load_mw", "season_sin", "season_cos", "hour"]

months = sorted(df["year_month"].unique())
BURN_IN_MONTHS = 2  # need at least this much history before the first fold

all_preds = []

for i in range(BURN_IN_MONTHS, len(months)):
    test_month = months[i]
    train_mask = df["year_month"] < test_month
    test_mask = df["year_month"] == test_month

    # Explicit leakage check: every training timestamp must be strictly
    # before every test timestamp. This is the check that actually matters
    # for the "no leakage" claim — not a comment, an assertion.
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

# --- Overall out-of-sample metrics (walk-forward, not a single split) ---
from sklearn.metrics import mean_squared_error, r2_score
wf_r2 = r2_score(backtest["dart_spread"], backtest["predicted"])
wf_rmse = np.sqrt(mean_squared_error(backtest["dart_spread"], backtest["predicted"]))

print(f"\n=== Walk-forward out-of-sample (all folds combined) ===")
print(f"Total OOS hours: {len(backtest)}")
print(f"R-squared: {wf_r2:.4f}")
print(f"RMSE:      ${wf_rmse:.3f}/MWh")

# --- Simple signal + toy backtest ---
# Signal: trade in the direction of the predicted spread sign.
# PnL assumption (STATED, not hidden): assumes a 1 MW notional position
# capturing the full realized spread when direction is correct, and
# losing the full realized spread when direction is wrong. This ignores
# transaction costs, bid-ask spread, capacity/volume constraints, and the
# fact that real DART trading requires submitting actual MW quantities
# into the day-ahead market, not just a directional bet. This is a
# DIRECTIONAL SKILL check, not a trading strategy backtest.
backtest["signal"] = np.sign(backtest["predicted"])
backtest["pnl"] = backtest["signal"] * backtest["dart_spread"]
backtest["correct_direction"] = (
    np.sign(backtest["predicted"]) == np.sign(backtest["dart_spread"])
)

hit_rate = backtest["correct_direction"].mean()
total_pnl = backtest["pnl"].sum()
avg_pnl_per_hour = backtest["pnl"].mean()
sharpe_like = backtest["pnl"].mean() / backtest["pnl"].std() * np.sqrt(24 * 365)  # annualized, hourly data

print(f"\n=== Toy directional backtest (no transaction costs, stated assumptions above) ===")
print(f"Hit rate (correct sign): {hit_rate*100:.1f}%")
print(f"Total PnL (1 MW notional, sum over {len(backtest)} hours): ${total_pnl:,.2f}")
print(f"Avg PnL per hour: ${avg_pnl_per_hour:.3f}")
print(f"Annualized Sharpe-like ratio (hourly, no costs): {sharpe_like:.2f}")

# Compare hit rate to a naive always-long or coin-flip baseline
naive_hit_rate_long = (backtest["dart_spread"] > 0).mean()
print(f"\nBaseline — always predict positive spread: {naive_hit_rate_long*100:.1f}% hit rate")
print(f"Baseline — coin flip: 50.0% hit rate")

backtest.to_csv("/home/claude/ercot_project/outputs/walk_forward_backtest.csv", index=False)
print("\nSaved full walk-forward results to outputs/walk_forward_backtest.csv")
