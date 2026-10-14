"""
SHAP values for the LightGBM model — a real step up from built-in feature
importance. Built-in importance (used in model_lgbm.py) only says "how
often was this feature used to split" — it doesn't say in which DIRECTION
a feature pushes the prediction, or whether that effect is consistent or
depends on the feature's value. SHAP answers both.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

FEATURES = ["forecast_error_mw", "net_load_mw", "season_sin", "season_cos", "hour"]
X = df[FEATURES].copy()
X["hour"] = X["hour"].astype("category")
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train = y.iloc[:split_idx]

model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=30, random_state=42, verbosity=-1,
)
model.fit(X_train, y_train, categorical_feature=["hour"])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP value per feature = global importance, but SIGNED
# and in the same units as the prediction ($/MWh), unlike split-count
# importance.
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=FEATURES
).sort_values(ascending=False)
print("=== Mean |SHAP value| per feature (average impact on predicted spread, $/MWh) ===")
print(mean_abs_shap)

# --- Does SHAP confirm the kink? ---
# Plot SHAP value vs feature value for forecast_error_mw — if the kink is
# real, this should show the same accelerating-slope pattern as the probe
# curve in model_lgbm.py, from a completely different method (SHAP
# decomposes actual predictions on real data points, not a synthetic probe
# grid), which is a meaningful cross-check.
fe_idx = FEATURES.index("forecast_error_mw")
fe_values = X_test["forecast_error_mw"].values
fe_shap = shap_values[:, fe_idx]

BLUE = "#2a78d6"
MUTED = "#898781"
mpl.rcParams.update({"axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
                      "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                      "savefig.facecolor": "#fcfcfb"})

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(fe_values, fe_shap, s=6, alpha=0.3, color=BLUE, edgecolors="none")
ax.axvline(1200, color=MUTED, lw=1, linestyle=":", label="Kink threshold (1,200 MW)")
ax.set_xlabel("forecast_error_mw (actual test-set values)")
ax.set_ylabel("SHAP value (impact on predicted spread, $/MWh)")
ax.set_title("SYNTHETIC DATA — SHAP dependence: cross-check on the nonlinear kink", fontsize=11)
ax.legend(frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig("/home/claude/ercot_project/outputs/charts/05_shap_dependence.png", dpi=150)
plt.close()
print("\nSaved SHAP dependence chart to outputs/charts/05_shap_dependence.png")

# Quantify the cross-check: slope before vs after threshold, same test as
# the probe curve, but on SHAP values from real (held-out) data points
# instead of a synthetic probe grid.
before = fe_shap[fe_values < 1200]
after_mask = fe_values > 3000
after = fe_shap[after_mask]
if len(before) > 10 and len(after) > 10:
    slope_before = np.polyfit(fe_values[fe_values < 1200], before, 1)[0]
    slope_after = np.polyfit(fe_values[after_mask], after, 1)[0]
    print(f"\nSHAP-based slope check (independent of the earlier probe-curve check):")
    print(f"  Slope below 1,200 MW: {slope_before:.5f}")
    print(f"  Slope above 3,000 MW: {slope_after:.5f}")
    print(f"  Confirms steeper slope past threshold: {slope_after < slope_before}")

shap_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in FEATURES])
shap_df.to_csv("/home/claude/ercot_project/outputs/shap_values.csv", index=False)
print("\nSaved raw SHAP values to outputs/shap_values.csv")
