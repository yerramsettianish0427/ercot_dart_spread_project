"""
OLS baseline refit with Newey-West (HAC) standard errors, compared against
plain OLS standard errors. Ljung-Box found autocorrelation at 24h, so the
plain SEs are likely too small. HAC doesn't change the coefficients.

maxlags = 24 to cover one daily cycle, same lag as the Ljung-Box test.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")
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

model_naive = sm.OLS(y_train, X_train).fit()
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
print("=== Plain vs HAC standard errors (key variables) ===")
print(compare.loc[key_vars].to_string(float_format=lambda x: f"{x:.6f}"))

fe_row = compare.loc["forecast_error_mw"]
print(f"\nforecast_error_mw coefficient: {fe_row['coefficient']:.6f}")
print(f"  Plain SE: {fe_row['naive_SE']:.6f}  ->  p={fe_row['naive_p']:.4g}")
print(f"  HAC SE:   {fe_row['HAC_SE']:.6f}  ->  p={fe_row['HAC_p']:.4g}")
print(f"  SE inflated by {fe_row['SE_inflation_pct']:.1f}% after correcting for autocorrelation")
print(f"\nStill significant at 5% after HAC: {fe_row['HAC_p'] < 0.05}")

compare.to_csv(os.path.join(ROOT, "outputs", "ols_hac_comparison.csv"))
print("\nSaved to outputs/ols_hac_comparison.csv")
