"""
LightGBM on the same target and same 80/20 split as model_ols.py.
Question: does a nonlinear model beat OLS out of sample?

Hour is passed as a categorical feature instead of dummies since trees
don't need one-hot encoding.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

FEATURES = ["forecast_error_mw", "net_load_mw", "season_sin", "season_cos", "hour"]
CATEGORICAL = ["hour"]

X = df[FEATURES].copy()
X["hour"] = X["hour"].astype("category")
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# kept shallow on purpose, 9600 rows is small for boosting
model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=30,
    random_state=42,
    verbosity=-1,
)
model.fit(
    X_train, y_train,
    categorical_feature=CATEGORICAL,
)

y_pred_test = model.predict(X_test)
oos_r2 = r2_score(y_test, y_pred_test)
oos_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
oos_mae = mean_absolute_error(y_test, y_pred_test)

print("=== LightGBM out-of-sample (same test set as OLS) ===")
print(f"R-squared (OOS): {oos_r2:.4f}")
print(f"RMSE (OOS):       ${oos_rmse:.3f}/MWh")
print(f"MAE (OOS):        ${oos_mae:.3f}/MWh")

ols_results = pd.read_csv(os.path.join(ROOT, "outputs", "model_comparison_ols.csv"))
ols_rmse = ols_results["oos_rmse"].iloc[0]
ols_r2 = ols_results["oos_r2"].iloc[0]

print(f"\n=== vs OLS ===")
print(f"OLS       RMSE: ${ols_rmse:.3f}  |  R²: {ols_r2:.4f}")
print(f"LightGBM  RMSE: ${oos_rmse:.3f}  |  R²: {oos_r2:.4f}")
improvement = (1 - oos_rmse / ols_rmse) * 100
print(f"LightGBM improvement over OLS: {improvement:.1f}%")
if improvement < 5:
    print("Small improvement, OLS is probably the better choice here.")

importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\nFeature importances:\n{importance}")

# Probe: vary forecast_error_mw with everything else fixed and look at the
# predicted curve. The kink should show up as a change in slope.
probe = pd.DataFrame({
    "forecast_error_mw": np.linspace(df["forecast_error_mw"].min(), df["forecast_error_mw"].max(), 200),
    "net_load_mw": df["net_load_mw"].median(),
    "season_sin": 0.0,
    "season_cos": 1.0,
    "hour": 12,
})
probe["hour"] = probe["hour"].astype("category").cat.set_categories(X["hour"].cat.categories)
probe["predicted_spread"] = model.predict(probe[FEATURES])
probe.to_csv(os.path.join(ROOT, "outputs", "lgbm_kink_probe.csv"), index=False)
print("\nSaved nonlinearity probe to outputs/lgbm_kink_probe.csv")

results = pd.DataFrame({
    "model": ["LightGBM"],
    "oos_r2": [oos_r2],
    "oos_rmse": [oos_rmse],
    "oos_mae": [oos_mae],
})
results.to_csv(os.path.join(ROOT, "outputs", "model_comparison_lgbm.csv"), index=False)
print("Saved results to outputs/model_comparison_lgbm.csv")
