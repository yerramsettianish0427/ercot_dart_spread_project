"""
GARCH(1,1) on the OLS residuals: does volatility cluster over time (calm
periods and chaotic periods come in streaks), rather than being constant?

IMPORTANT CAVEAT (stated before running, not after seeing the result):
the synthetic data generator draws its noise and "large error regime"
independently each hour (i.i.d.) — it does NOT build in any real
volatility clustering. So this test, on THIS data, may honestly come back
null (no significant clustering), which would be the CORRECT result
given how the data was built, not a failure of the method. Real ERCOT
data would likely show genuine clustering (storms, grid stress events
span multiple hours), and this script is written to be rerun unchanged
on real residuals later.

GARCH(1,1): sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
  alpha: how much yesterday's shock affects today's volatility
  beta:  how much volatility itself persists
  alpha + beta close to 1 -> highly persistent volatility (real markets
  often look like this); close to 0 -> no clustering, constant variance.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

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

ols = sm.OLS(y_train, X_train).fit()
residuals = ols.resid

# --- Step 1: test for ARCH effects BEFORE fitting GARCH ---
# This is the step the first version of this script skipped, and skipping
# it produced a misleading read: GARCH's optimizer can converge to
# alpha~0, beta~1 on data with NO real clustering (a known degenerate
# fit on the likelihood's flat ridge), and beta's p-value alone looks
# "significant" if you don't check alpha too. Engle's LM test is the
# correct, standard way to check whether GARCH is even warranted.
lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=24)
print(f"=== Engle ARCH-LM test (run BEFORE trusting any GARCH fit) ===")
print(f"LM statistic: {lm_stat:.4f}, p-value: {lm_pvalue:.4f}")
arch_effects_present = lm_pvalue < 0.05

# --- Step 2: fit GARCH regardless, but interpret it through the LM result ---
garch = arch_model(residuals, vol="Garch", p=1, q=1, mean="Zero", rescale=True)
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

alpha = garch_fit.params.get("alpha[1]", np.nan)
beta = garch_fit.params.get("beta[1]", np.nan)
alpha_p = garch_fit.pvalues.get("alpha[1]", np.nan)
beta_p = garch_fit.pvalues.get("beta[1]", np.nan)

print(f"\n=== Interpretation ===")
print(f"ARCH-LM test p-value: {lm_pvalue:.4f} -> "
      f"{'ARCH effects present' if arch_effects_present else 'NO significant ARCH effects'}")
print(f"alpha (shock impact):          {alpha:.4f}  (p={alpha_p:.4g})")
print(f"beta (volatility persistence): {beta:.4f}  (p={beta_p:.4g})")

if not arch_effects_present:
    print("\nResult: the ARCH-LM test — the correct test to trust here — finds")
    print("NO significant volatility clustering. This is the EXPECTED, HONEST")
    print("result for this synthetic dataset: the generator draws noise and")
    print("regime independently each hour, so there's genuinely nothing for")
    print("GARCH to find.")
    if alpha_p > 0.05 and beta_p < 0.05:
        print("\nNote on the GARCH fit itself: alpha is statistically")
        print("indistinguishable from zero while beta alone looks")
        print("'significant' — this specific pattern (alpha~0, beta~1) is a")
        print("known GARCH estimation degeneracy on data with no real ARCH")
        print("effect, NOT evidence of clustering. Reporting beta's p-value")
        print("in isolation here would be a methodological error.")
    print("\nOn real ERCOT data, where weather fronts and grid-stress events")
    print("genuinely span multiple hours, re-running this same ARCH-LM test")
    print("first would likely find real, significant clustering — at which")
    print("point the GARCH fit above would become trustworthy to interpret.")
else:
    print("\nResult: ARCH-LM confirms real volatility clustering. The GARCH")
    print("alpha/beta estimates above are safe to interpret directly.")

# Save conditional volatility series for charting
cond_vol = pd.DataFrame({
    "timestamp": df["timestamp"].iloc[:split_idx].values,
    "residual": residuals.values,
    "conditional_volatility": garch_fit.conditional_volatility.values,
})
cond_vol.to_csv("/home/claude/ercot_project/outputs/garch_conditional_volatility.csv", index=False)
print("\nSaved conditional volatility series to outputs/garch_conditional_volatility.csv")
