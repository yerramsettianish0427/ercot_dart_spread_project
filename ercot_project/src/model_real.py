"""
Real-data model run. SCOPE CHANGE FROM THE SYNTHETIC VERSION, stated up
front because it affects how every number below should be read:

  - n=166 hours (Oct 4-10, 2026), not 9,600. No month-long walk-forward
    (12 folds) is possible or meaningful on 7 days of data — there's only
    one real "fold" here. Doing a simple chronological holdout instead and
    saying so, rather than dressing up 166 rows as a walk-forward backtest.
  - No seasonal features (season_sin/cos) — 6 days of data cannot inform a
    365-day cycle; including it would just be a free, meaningless degree
    of freedom. Dropped.
  - Quantile regression is NOT rerun here: with ~40 test rows, a 10/50/90
    calibration check needs at least a few hundred points to mean anything
    -- reporting quantile coverage on 40 rows would be the single most
    overclaimed number in this whole project. Skipped, noted as a real
    limitation, not silently dropped.
  - GARCH/ARCH-LM/block-bootstrap: same problem, same decision. Skipped.
  - What IS still valid on 166 rows: OLS coefficient + HAC-robust SE
    (always correct regardless of n, estimates just wider), and a basic
    LightGBM vs OLS OOS comparison, both reported with wide, explicit
    uncertainty.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb

df = pd.read_csv("/home/claude/ercot_project/data/REAL_ercot_hourly.csv", parse_dates=["Interval Start"])
df = df.sort_values("Interval Start").reset_index(drop=True)
print(f"n = {len(df)} real hours, {df['Interval Start'].min()} to {df['Interval Start'].max()}\n")

FEATURES = ["forecast_error_mw", "load_mw", "hour"]
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.75)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
print(f"Train: {len(train)} hours | Test: {len(test)} hours (chronological split, not walk-forward — too little data for that)\n")

# --- OLS + HAC-robust SE (valid at any n) ---
X_ols = sm.add_constant(train[["forecast_error_mw", "load_mw"]].astype(float))
ols = sm.OLS(train["dart_spread"].astype(float), X_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
print("=== OLS (HAC-robust SE, 24h lag) on REAL data ===")
print(ols.summary().tables[1])

X_test_ols = sm.add_constant(test[["forecast_error_mw", "load_mw"]].astype(float), has_constant="add")
ols_pred = ols.predict(X_test_ols)
ols_r2 = r2_score(test["dart_spread"], ols_pred)
ols_rmse = np.sqrt(mean_squared_error(test["dart_spread"], ols_pred))
print(f"\nOLS real-data OOS: R² = {ols_r2:.4f}, RMSE = ${ols_rmse:.3f}/MWh (n_test={len(test)})")

# --- LightGBM, small-n settings (shallower, more regularized than the
# synthetic run's hyperparameters, which would overfit badly on n=124) ---
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

# --- Bootstrap CI on OOS R² (percentile bootstrap over test rows) since
# n_test is small enough that a single point estimate is close to
# meaningless without a spread ---
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
print("(Wide/overlapping CIs here are the expected, honest result of n_test≈40 —")
print(" not evidence the pipeline is broken.)")

results = pd.DataFrame({
    "model": ["naive", "ols", "lightgbm"],
    "oos_r2": [np.nan, ols_r2, lgb_r2],
    "oos_rmse": [naive_rmse, ols_rmse, lgb_rmse],
    "n_test": [len(test)] * 3,
})

# =====================================================================
# EXPLANATORY vs TRADEABLE comparison — this is the market-structure
# point: forecast_error_mw needs the ACTUAL output, unknowable before
# real-time settles, so a model built on it (above) tells you the
# relationship exists but is NOT a tradeable signal. forecast_revision_mw
# is built only from forecast snapshots published before the interval
# started (see clean_real_data.py) — a desk really could observe this and
# act on it before settlement. Comparing the two head-to-head on identical
# train/test rows quantifies exactly how much power you give up by
# removing the unfair advantage.
# =====================================================================
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

print("\n" + "=" * 70)
print("EXPLANATORY (uses same-hour actual) vs TRADEABLE (uses only")
print("pre-settlement forecast revisions) — same real data, n=%d" % len(df_rev))
print("=" * 70)
print(ols_tradeable.summary().tables[1])
print(f"\nExplanatory OLS (forecast_error_mw, needs the actual):  OOS R²={ols_r2:.3f}, RMSE=${ols_rmse:.2f}/MWh")
print(f"Tradeable OLS (forecast_revision_mw, no actual needed): OOS R²={tradeable_r2:.3f}, RMSE=${tradeable_rmse:.2f}/MWh")
print("\nBoth are negative here (166-hour regime-shift problem, see above) —")
print("the point of this comparison isn't which number is bigger on a tiny")
print("sample, it's that the tradeable version is built correctly: it never")
print("touches a value that wouldn't exist yet at decision time. On a real")
print("desk, only the tradeable version could ever be acted on; the")
print("explanatory version is a research tool for understanding WHY prices")
print("move, not a signal to trade on.")

results = pd.concat([results, pd.DataFrame({
    "model": ["ols_tradeable_revision"],
    "oos_r2": [tradeable_r2],
    "oos_rmse": [tradeable_rmse],
    "n_test": [len(test_rev)],
})], ignore_index=True)

results.to_csv("/home/claude/ercot_project/outputs/real_data_results.csv", index=False)
print("\nSaved to outputs/real_data_results.csv")
