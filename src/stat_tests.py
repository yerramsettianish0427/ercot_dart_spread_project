"""
ADF stationarity test and Ljung-Box autocorrelation test on the DART spread.
A unit root would make regression coefficients on it potentially spurious.
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv"), parse_dates=["timestamp"])
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
    print("\nReject H0 at 5%: stationary, no differencing needed.")
else:
    print("\nFail to reject H0: may be non-stationary, consider differencing.")

lb_result = acorr_ljungbox(spread, lags=[24], return_df=True)
print(f"\n=== Ljung-Box test for autocorrelation (lag=24 hours) ===")
print(lb_result)
if lb_result["lb_pvalue"].iloc[0] < 0.05:
    print("\nSignificant autocorrelation at 24h, so plain OLS standard errors")
    print("will be too small. See model_ols_hac.py.")
