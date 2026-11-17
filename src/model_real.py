"""
Models on the real 166-hour window (Oct 4-10, 2026).

Differences from the synthetic runs:
  - One chronological 75/25 split. A monthly walk-forward isn't possible on
    a week of data.
  - No season features, 6 days can't say anything about a yearly cycle.
  - Quantile regression, GARCH and the block bootstrap are skipped. ~40 test
    rows is too few for any of them.
  - OLS with HAC SEs and a small LightGBM, with bootstrap CIs on OOS R².
  - At the end, the explanatory feature (forecast_error_mw, needs the
    actual) vs the tradeable one (forecast_revision_mw, known beforehand).
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT, "data", "REAL_ercot_hourly.csv"), parse_dates=["Interval Start"])
df = df.sort_values("Interval Start").reset_index(drop=True)
print(f"n = {len(df)} real hours, {df['Interval Start'].min()} to {df['Interval Start'].max()}\n")

FEATURES = ["forecast_error_mw", "load_mw", "hour"]
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.75)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
print(f"Train: {len(train)} hours | Test: {len(test)} hours (chronological split)\n")

X_ols = sm.add_constant(train[["forecast_error_mw", "load_mw"]].astype(float))
ols = sm.OLS(train["dart_spread"].astype(float), X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
print("=== OLS (HAC SE, 24h lag) on real data ===")
print(ols.summary().tables[1])

X_test_ols = sm.add_constant(test[["forecast_error_mw", "load_mw"]].astype(float), has_constant="add")
ols_pred = ols.predict(X_test_ols)
ols_r2 = r2_score(test["dart_spread"], ols_pred)
ols_rmse = np.sqrt(mean_squared_error(test["dart_spread"], ols_pred))
print(f"\nOLS real-data OOS: R² = {ols_r2:.4f}, RMSE = ${ols_rmse:.3f}/MWh (n_test={len(test)})")

# much smaller than the synthetic settings, n_train is 124
X_train_lgb = train[FEATURES].copy()
X_train_lgb["hour"] = X_train_lgb["hour"].astype("category")
X_test_lgb = test[FEATURES].copy()
X_test_lgb["hour"] = X_test_lgb["hour"].astype("category")

model = lgb.LGBMRegressor(
    n_estimators=50, max_depth=2, num_leaves=4, learning_rate=0.05,
    min_child_samples=10, random_state=42, verbosity=-1,
)
model.fit(X_train_lgb, train["dart_spread"].astype(float), categorical_feature=["hour"])
lgb_pred = model.predict(X_test_lgb)
lgb_r2 = r2_score(test["dart_spread"], lgb_pred)
lgb_rmse = np.sqrt(mean_squared_error(test["dart_spread"], lgb_pred))
print(f"LightGBM real-data OOS: R² = {lgb_r2:.4f}, RMSE = ${lgb_rmse:.3f}/MWh (n_test={len(test)})")

naive_pred = np.full(len(test), train["dart_spread"].mean())
naive_rmse = np.sqrt(mean_squared_error(test["dart_spread"], naive_pred))
print(f"Naive (predict train mean): RMSE = ${naive_rmse:.3f}/MWh")

# percentile bootstrap over test rows
np.random.seed(42)
n_boot = 2000
boot_r2_ols, boot_r2_lgb = [], []
test_idx = np.arange(len(test))
for _ in range(n_boot):
    s = np.random.choice(test_idx, size=len(test_idx), replace=True)
    y_s = test["dart_spread"].values[s]
    if y_s.std() == 0:
        continue
    boot_r2_ols.append(r2_score(y_s, ols_pred.values[s]))
    boot_r2_lgb.append(r2_score(y_s, lgb_pred[s]))

print(f"\nOLS OOS R² 95% bootstrap CI:      [{np.percentile(boot_r2_ols,2.5):.3f}, {np.percentile(boot_r2_ols,97.5):.3f}]")
print(f"LightGBM OOS R² 95% bootstrap CI: [{np.percentile(boot_r2_lgb,2.5):.3f}, {np.percentile(boot_r2_lgb,97.5):.3f}]")

results = pd.DataFrame({
    "model": ["naive", "ols", "lightgbm"],
    "oos_r2": [np.nan, ols_r2, lgb_r2],
    "oos_rmse": [naive_rmse, ols_rmse, lgb_rmse],
    "n_test": [len(test)] * 3,
})

# Explanatory vs tradeable. forecast_error_mw needs the same-hour actual,
# which nobody has before real-time settles. forecast_revision_mw only uses
# forecasts published before the hour, so it could actually be traded.
df_rev = df.dropna(subset=["forecast_revision_mw"]).reset_index(drop=True)
split_idx_rev = int(len(df_rev) * 0.75)
train_rev, test_rev = df_rev.iloc[:split_idx_rev], df_rev.iloc[split_idx_rev:]

X_tr_tradeable = sm.add_constant(train_rev[["forecast_revision_mw", "load_mw"]].astype(float))
ols_tradeable = sm.OLS(train_rev["dart_spread"].astype(float), X_tr_tradeable).fit(
    cov_type="HAC", cov_kwds={"maxlags": 24}
)
X_te_tradeable = sm.add_constant(test_rev[["forecast_revision_mw", "load_mw"]].astype(float), has_constant="add")
tradeable_pred = ols_tradeable.predict(X_te_tradeable)
tradeable_r2 = r2_score(test_rev["dart_spread"], tradeable_pred)
tradeable_rmse = np.sqrt(mean_squared_error(test_rev["dart_spread"], tradeable_pred))

print("\n=== Explanatory vs tradeable, n=%d ===" % len(df_rev))
print(ols_tradeable.summary().tables[1])
print(f"\nExplanatory OLS (forecast_error_mw, needs the actual):  OOS R²={ols_r2:.3f}, RMSE=${ols_rmse:.2f}/MWh")
print(f"Tradeable OLS (forecast_revision_mw, no actual needed): OOS R²={tradeable_r2:.3f}, RMSE=${tradeable_rmse:.2f}/MWh")

results = pd.concat([results, pd.DataFrame({
    "model": ["ols_tradeable_revision"],
    "oos_r2": [tradeable_r2],
    "oos_rmse": [tradeable_rmse],
    "n_test": [len(test_rev)],
})], ignore_index=True)

results.to_csv(os.path.join(ROOT, "outputs", "real_data_results.csv"), index=False)
print("\nSaved to outputs/real_data_results.csv")
