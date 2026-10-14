"""
Re-fit the OLS baseline with Newey-West (HAC) standard errors and compare
directly against the naive (non-robust) standard errors from model_ols.py.

Why this matters: the Ljung-Box test (stat_tests.py) found real
autocorrelation in the spread at a 24-hour lag. Plain OLS assumes
independent errors — when that's false, the standard errors (and
therefore the p-values and confidence intervals) are wrong, usually too
small, which can make a coefficient look more significant than it is.
HAC correction doesn't change the point estimates (the coefficients
themselves are unchanged) — it only fixes the uncertainty around them.

maxlags choice: 24 hours (one full day), matching the lag used in the
Ljung-Box test — the idea is to allow for autocorrelation up to one full
daily cycle, which is the natural periodicity in this data (diurnal load
and wind patterns).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

hour_dummies = pd.get_dummies(df["hour"], prefix="hour", drop_first=True)
X = pd.concat([
    df[["forecast_error_mw", "net_load_mw", "season_sin", "season_cos"]],
    hour_dummies,
], axis=1)
X = sm.add_constant(X).astype(float)
y = df["dart_spread"].astype(float)

split_idx = int(len(df) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

# --- Naive (non-robust) SEs — what model_ols.py originally reported ---
model_naive = sm.OLS(y_train, X_train).fit()

# --- HAC (Newey-West) robust SEs ---
model_hac = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": 24})

compare = pd.DataFrame({
    "coefficient": model_naive.params,
    "naive_SE": model_naive.bse,
    "naive_p": model_naive.pvalues,
    "HAC_SE": model_hac.bse,
    "HAC_p": model_hac.pvalues,
})
compare["SE_inflation_pct"] = (compare["HAC_SE"] / compare["naive_SE"] - 1) * 100

key_vars = ["const", "forecast_error_mw", "net_load_mw", "season_sin", "season_cos"]
print("=== Naive vs HAC-robust standard errors (key variables) ===")
print(compare.loc[key_vars].to_string(float_format=lambda x: f"{x:.6f}"))

print(f"\n=== Interpretation ===")
fe_row = compare.loc["forecast_error_mw"]
print(f"forecast_error_mw coefficient: {fe_row['coefficient']:.6f} (unchanged by HAC — point estimate is the same)")
print(f"  Naive SE: {fe_row['naive_SE']:.6f}  ->  p={fe_row['naive_p']:.4g}")
print(f"  HAC SE:   {fe_row['HAC_SE']:.6f}  ->  p={fe_row['HAC_p']:.4g}")
print(f"  SE inflated by {fe_row['SE_inflation_pct']:.1f}% after correcting for autocorrelation")

still_significant = fe_row["HAC_p"] < 0.05
print(f"\nStill statistically significant at 5% after HAC correction: {still_significant}")
if still_significant:
    print("This is the honest version of the claim: even after properly")
    print("accounting for autocorrelated errors, the forecast-error effect")
    print("holds up. That's a stronger statement than the naive p-value alone.")

compare.to_csv("/home/claude/ercot_project/outputs/ols_hac_comparison.csv")
print("\nSaved to outputs/ols_hac_comparison.csv")
