"""
ARCH-LM test and GARCH(1,1) on the OLS residuals: does volatility cluster?

The synthetic generator draws noise independently each hour, so there
shouldn't be any clustering to find here.

GARCH(1,1): sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

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

ols = sm.OLS(y_train, X_train).fit()
residuals = ols.resid

# Run ARCH-LM first. My first version skipped it and read beta's p-value
# as evidence of clustering, but GARCH can converge to alpha~0, beta~1 on
# data with no ARCH effect at all.
lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=24)
print(f"=== Engle ARCH-LM test ===")
print(f"LM statistic: {lm_stat:.4f}, p-value: {lm_pvalue:.4f}")
arch_effects_present = lm_pvalue < 0.05

garch = arch_model(residuals, vol="Garch", p=1, q=1, mean="Zero", rescale=True)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

alpha = garch_fit.params.get("alpha[1]", np.nan)
beta = garch_fit.params.get("beta[1]", np.nan)
alpha_p = garch_fit.pvalues.get("alpha[1]", np.nan)
beta_p = garch_fit.pvalues.get("beta[1]", np.nan)

print(f"\nARCH-LM p-value: {lm_pvalue:.4f} -> "
      f"{'ARCH effects present' if arch_effects_present else 'no significant ARCH effects'}")
print(f"alpha (shock impact):          {alpha:.4f}  (p={alpha_p:.4g})")
print(f"beta (volatility persistence): {beta:.4f}  (p={beta_p:.4g})")

if not arch_effects_present and alpha_p > 0.05 and beta_p < 0.05:
    print("\nalpha ~ 0 with a 'significant' beta is the degenerate fit, not clustering.")

cond_vol = pd.DataFrame({
    "timestamp": df["timestamp"].iloc[:split_idx].values,
    "residual": residuals.values,
    "conditional_volatility": garch_fit.conditional_volatility.values,
})
cond_vol.to_csv(os.path.join(ROOT, "outputs", "garch_conditional_volatility.csv"), index=False)
print("\nSaved conditional volatility series to outputs/garch_conditional_volatility.csv")
