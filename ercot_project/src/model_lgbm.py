"""
LightGBM: same target and same chronological 80/20 split as OLS, so the
comparison is apples-to-apples. The only question this script answers:
does a nonlinear model meaningfully beat OLS out-of-sample? If not, that's
a legitimate result to report — "we tried more complexity and it didn't
help" is more credible than forcing a fancier model to "win."

Feature treatment differs slightly from OLS on purpose: trees don't need
one-hot dummies for hour-of-day, so hour is passed as a native categorical
feature. This is a deliberate, stated modeling choice, not an inconsistency.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"
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

# Deliberately shallow/regularized — this is a small dataset (9600 rows) by
# gradient-boosting standards, and the point is to test one specific
# hypothesis (a nonlinear kink exists), not to squeeze out every bit of
# fit. An over-tuned model here would be a red flag in an interview, not
# a strength.
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

# --- Compare directly against OLS ---
ols_results = pd.read_csv("/home/claude/ercot_project/outputs/model_comparison_ols.csv")
ols_rmse = ols_results["oos_rmse"].iloc[0]
ols_r2 = ols_results["oos_r2"].iloc[0]

print(f"\n=== Head-to-head vs OLS ===")
print(f"OLS       RMSE: ${ols_rmse:.3f}  |  R²: {ols_r2:.4f}")
print(f"LightGBM  RMSE: ${oos_rmse:.3f}  |  R²: {oos_r2:.4f}")
improvement = (1 - oos_rmse / ols_rmse) * 100
print(f"LightGBM improvement over OLS: {improvement:.1f}%")
if improvement < 5:
    print("NOTE: improvement is small. Given the added complexity, OLS may")
    print("be the more defensible choice for production use — complexity")
    print("should earn its place, not be assumed to help.")

# --- Feature importance ---
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\nFeature importances:\n{importance}")

# --- Does LightGBM actually capture the nonlinear kink? ---
# Hold net_load, season, hour at their median/mode, vary forecast_error_mw
# across its observed range, and look at the predicted spread curve. A
# kink should show up as a change in slope, which OLS (linear by
# construction) cannot represent.
probe = pd.DataFrame({
    "forecast_error_mw": np.linspace(df["forecast_error_mw"].min(), df["forecast_error_mw"].max(), 200),
    "net_load_mw": df["net_load_mw"].median(),
    "season_sin": 0.0,
    "season_cos": 1.0,
    "hour": 12,
})
probe["hour"] = probe["hour"].astype("category").cat.set_categories(X["hour"].cat.categories)
probe["predicted_spread"] = model.predict(probe[FEATURES])
probe.to_csv("/home/claude/ercot_project/outputs/lgbm_kink_probe.csv", index=False)
print("\nSaved nonlinearity probe to outputs/lgbm_kink_probe.csv (for charting)")

results = pd.DataFrame({
    "model": ["LightGBM"],
    "oos_r2": [oos_r2],
    "oos_rmse": [oos_rmse],
    "oos_mae": [oos_mae],
})
results.to_csv("/home/claude/ercot_project/outputs/model_comparison_lgbm.csv", index=False)
print("Saved results to outputs/model_comparison_lgbm.csv")
