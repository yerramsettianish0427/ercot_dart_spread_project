"""
OLS baseline: DART_spread ~ forecast_error + net_load + hour-of-day + season.
Single chronological 80/20 split. Walk-forward is in model_backtest.py.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
assert df["synthetic_data_flag"].all(), "Expected all rows to be flagged synthetic"

hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=True)

# Sin/cos of day-of-year instead of month dummies. With month dummies the
# first run broke: the train split ended mid-November, so the December
# column was all zeros and the design matrix was rank-deficient.
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

X = pd.concat([
    df[["forecast_error_mw", "net_load_mw", "season_sin", "season_cos"]],
    hour_dummies,
], axis=1)
X = sm.add_constant(X)
X = X.astype(float)
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} rows ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[split_idx-1]})")
print(f"Test:  {len(X_test)} rows ({df['timestamp'].iloc[split_idx]} to {df['timestamp'].iloc[-1]})")

model = sm.OLS(y_train, X_train).fit()

print("\n=== In-sample fit (train only) ===")
print(f"R-squared: {model.rsquared:.4f}")
print(f"\nforecast_error_mw coefficient: {model.params['forecast_error_mw']:.6f}")
print(f"  (a 1000 MW forecast miss moves the DART spread by "
      f"${model.params['forecast_error_mw']*1000:.2f}/MWh, other things fixed)")
print(f"  p-value: {model.pvalues['forecast_error_mw']:.4g}")
print(f"\nnet_load_mw coefficient: {model.params['net_load_mw']:.6f}")
print(f"  p-value: {model.pvalues['net_load_mw']:.4g}")

y_pred_test = model.predict(X_test)
oos_r2 = r2_score(y_test, y_pred_test)
oos_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
oos_mae = mean_absolute_error(y_test, y_pred_test)

naive_pred = np.full_like(y_test, y_train.mean(), dtype=float)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

print("\n=== Out-of-sample (test set) ===")
print(f"R-squared (OOS):  {oos_r2:.4f}")
print(f"RMSE (OOS):        ${oos_rmse:.3f}/MWh")
print(f"MAE (OOS):         ${oos_mae:.3f}/MWh")
print(f"Naive RMSE (predict train mean): ${naive_rmse:.3f}/MWh")
print(f"Improvement over naive: {(1 - oos_rmse/naive_rmse)*100:.1f}%")

results = pd.DataFrame({
    "model": ["OLS"],
    "oos_r2": [oos_r2],
    "oos_rmse": [oos_rmse],
    "oos_mae": [oos_mae],
    "naive_rmse": [naive_rmse],
})
results.to_csv(os.path.join(ROOT, "outputs", "model_comparison_ols.csv"), index=False)
print("\nSaved results to outputs/model_comparison_ols.csv")
