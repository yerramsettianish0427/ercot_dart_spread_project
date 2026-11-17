"""
Quantile regression for the 10th/50th/90th percentile of the DART spread,
using LightGBM's quantile objective so the kink carries through.

Checked by calibration: about 90% of test outcomes should fall below q90.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

FEATURES = ["forecast_error_mw", "net_load_mw", "season_sin", "season_cos", "hour"]
X = df[FEATURES].copy()
X["hour"] = X["hour"].astype("category")
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

quantiles = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
preds = {}

for name, tau in quantiles.items():
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=tau,
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=30,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=["hour"])
    preds[name] = model.predict(X_test)
    print(f"Fitted {name} (tau={tau})")

results = pd.DataFrame(preds, index=X_test.index)
results["actual"] = y_test.values
results["timestamp"] = df["timestamp"].iloc[split_idx:].values

# separately fit quantile models can cross
violations = (
    (results["q10"] > results["q50"]) | (results["q50"] > results["q90"])
).sum()
print(f"\nMonotonicity violations (q10>q50 or q50>q90): {violations} / {len(results)} "
      f"({violations/len(results)*100:.2f}%)")

coverage_90 = (results["actual"] <= results["q90"]).mean()
coverage_10 = (results["actual"] <= results["q10"]).mean()
coverage_50 = (results["actual"] <= results["q50"]).mean()

print(f"\n=== Calibration (out-of-sample) ===")
print(f"Target: 10% of actuals below q10.  Actual: {coverage_10*100:.1f}%")
print(f"Target: 50% of actuals below q50.  Actual: {coverage_50*100:.1f}%")
print(f"Target: 90% of actuals below q90.  Actual: {coverage_90*100:.1f}%")

for label, target, actual in [("q10", 0.10, coverage_10), ("q50", 0.50, coverage_50), ("q90", 0.90, coverage_90)]:
    gap = abs(actual - target)
    flag = "OK" if gap < 0.05 else "MISCALIBRATED"
    print(f"  {label}: gap = {gap*100:.1f} pts -> {flag}")

results.to_csv(os.path.join(ROOT, "outputs", "quantile_predictions.csv"), index=False)
print("\nSaved predictions to outputs/quantile_predictions.csv")
