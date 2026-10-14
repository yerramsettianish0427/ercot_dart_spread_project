"""
OLS baseline: DART_spread ~ forecast_error + net_load + hour-of-day + month

This is deliberately the simplest model in the pipeline. Its job is to:
  1. Give an interpretable, signed coefficient on forecast_error (the
     headline "does a wind forecast miss move the spread, and by how much"
     number).
  2. Be the benchmark that LightGBM has to beat to justify its added
     complexity — if LightGBM doesn't meaningfully improve on out-of-sample
     error, OLS is the honest answer and that's a fine result to report.

NOTE: this script uses a single chronological 80/20 train/test split, not
proper walk-forward validation. Walk-forward is built in model_backtest.py
(Task 5). This script exists to validate the OLS specification in isolation
before it's reused inside the walk-forward loop.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
assert df["synthetic_data_flag"].all(), "Expected all rows to be flagged synthetic"

# --- Feature engineering ---
# Hour-of-day dummies (drop one to avoid perfect collinearity with intercept).
# All 24 hours appear every single day, so this is safe under any
# chronological split.
hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=True)

# Seasonality as continuous Fourier terms instead of month dummies.
# Month dummies are fragile under a chronological train/test split: if any
# calendar month is incompletely represented in train (e.g. train ends
# mid-November, so "December" never appears), that dummy column is all
# zeros in X_train and the design matrix becomes rank-deficient (exactly
# what happened on the first run of this script). Sin/cos of day-of-year
# varies smoothly and continuously, so there's no missing-category failure
# mode — this is also just a more standard way to encode annual
# seasonality than 11 dummy variables.
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

X = pd.concat([
    df[["forecast_error_mw", "net_load_mw", "season_sin", "season_cos"]],
    hour_dummies,
], axis=1)
X = sm.add_constant(X)
X = X.astype(float)
y = df["dart_spread"].astype(float)

# --- Chronological 80/20 split (NOT random — this is time series) ---
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} rows ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[split_idx-1]})")
print(f"Test:  {len(X_test)} rows ({df['timestamp'].iloc[split_idx]} to {df['timestamp'].iloc[-1]})")

# --- Fit ---
model = sm.OLS(y_train, X_train).fit()

print("\n=== In-sample fit (train only) ===")
print(f"R-squared: {model.rsquared:.4f}")
print(f"\nKey coefficient — forecast_error_mw: {model.params['forecast_error_mw']:.6f}")
print(f"  (interpretation: a 1000 MW forecast miss moves the DART spread by "
      f"${model.params['forecast_error_mw']*1000:.2f}/MWh, holding other things fixed)")
print(f"  p-value: {model.pvalues['forecast_error_mw']:.4g}")
print(f"\nKey coefficient — net_load_mw: {model.params['net_load_mw']:.6f}")
print(f"  p-value: {model.pvalues['net_load_mw']:.4g}")

# --- Out-of-sample evaluation (the number that actually matters) ---
y_pred_test = model.predict(X_test)
oos_r2 = r2_score(y_test, y_pred_test)
oos_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
oos_mae = mean_absolute_error(y_test, y_pred_test)

# Naive baseline: predict the mean of the train set for every test point.
# OLS has to beat THIS to be worth anything.
naive_pred = np.full_like(y_test, y_train.mean(), dtype=float)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

print("\n=== Out-of-sample (test set, never seen during fit) ===")
print(f"R-squared (OOS):  {oos_r2:.4f}")
print(f"RMSE (OOS):        ${oos_rmse:.3f}/MWh")
print(f"MAE (OOS):         ${oos_mae:.3f}/MWh")
print(f"Naive RMSE (predict train mean): ${naive_rmse:.3f}/MWh")
print(f"Improvement over naive: {(1 - oos_rmse/naive_rmse)*100:.1f}%")

# Save for comparison against LightGBM later
results = pd.DataFrame({
    "model": ["OLS"],
    "oos_r2": [oos_r2],
    "oos_rmse": [oos_rmse],
    "oos_mae": [oos_mae],
    "naive_rmse": [naive_rmse],
})
results.to_csv("/home/claude/ercot_project/outputs/model_comparison_ols.csv", index=False)
print("\nSaved results to outputs/model_comparison_ols.csv")
