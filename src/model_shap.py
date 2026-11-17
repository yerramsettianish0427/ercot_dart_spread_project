"""
SHAP values for the LightGBM model. Split-count importance doesn't show
direction or whether the effect changes with the feature value; SHAP does.
Also used as a second check on the kink, on real test points rather than
the probe grid in model_lgbm.py.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl

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
y_train = y.iloc[:split_idx]

model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
    min_child_samples=30, random_state=42, verbosity=-1,
)
model.fit(X_train, y_train, categorical_feature=["hour"])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), index=FEATURES
).sort_values(ascending=False)
print("=== Mean |SHAP value| per feature ($/MWh) ===")
print(mean_abs_shap)

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
ax.set_title("SYNTHETIC DATA: SHAP dependence for forecast error", fontsize=11)
ax.legend(frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "outputs", "charts", "05_shap_dependence.png"), dpi=150)
plt.close()
print("\nSaved SHAP dependence chart to outputs/charts/05_shap_dependence.png")

# slope of SHAP vs feature value below the threshold vs well above it
before = fe_shap[fe_values < 1200]
after_mask = fe_values > 3000
after = fe_shap[after_mask]
if len(before) > 10 and len(after) > 10:
    slope_before = np.polyfit(fe_values[fe_values < 1200], before, 1)[0]
    slope_after = np.polyfit(fe_values[after_mask], after, 1)[0]
    print(f"\nSHAP slope check:")
    print(f"  Slope below 1,200 MW: {slope_before:.5f}")
    print(f"  Slope above 3,000 MW: {slope_after:.5f}")
    print(f"  Steeper past threshold: {slope_after < slope_before}")

shap_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in FEATURES])
shap_df.to_csv(os.path.join(ROOT, "outputs", "shap_values.csv"), index=False)
print("\nSaved raw SHAP values to outputs/shap_values.csv")
