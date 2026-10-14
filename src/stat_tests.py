"""
Stationarity check on the DART spread series — a basic time-series hygiene
step that was skipped in the first pass. If the spread has a unit root
(is non-stationary), regression coefficients on it can be spurious —
the textbook example is two unrelated random walks appearing "correlated"
purely because both trend over time.

Augmented Dickey-Fuller test:
  H0: series has a unit root (non-stationary)
  H1: series is stationary
  Reject H0 (low p-value) -> stationary -> safe to model directly.
  Fail to reject -> would need differencing before modeling.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv", parse_dates=["timestamp"])
spread = df["dart_spread"].dropna()

result = adfuller(spread, autolag="AIC")

print("=== Augmented Dickey-Fuller test: DART spread ===")
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.6f}")
print(f"# lags used:   {result[2]}")
print(f"# observations: {result[3]}")
print("Critical values:")
for key, val in result[4].items():
    print(f"  {key}: {val:.4f}")

if result[1] < 0.05:
    print("\nResult: REJECT H0 at 5% level -> series is stationary.")
    print("Safe to model dart_spread directly without differencing.")
else:
    print("\nResult: FAIL TO REJECT H0 -> series may be non-stationary.")
    print("Regression coefficients on this series could be spurious;")
    print("would need differencing or detrending before modeling.")

# Also check: is there significant autocorrelation in the residuals a
# plain OLS would ignore? This motivates the Newey-West correction next.
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(spread, lags=[24], return_df=True)
print(f"\n=== Ljung-Box test for autocorrelation (lag=24 hours) ===")
print(lb_result)
if lb_result["lb_pvalue"].iloc[0] < 0.05:
    print("\nResult: significant autocorrelation detected at 24-hour lag.")
    print("Plain OLS standard errors will understate true uncertainty —")
    print("this is exactly what Newey-West (HAC) correction fixes.")
