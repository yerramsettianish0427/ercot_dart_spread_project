"""
*** SYNTHETIC DATA — NOT REAL ERCOT DATA ***

This generates fake-but-statistically-plausible hourly data standing in for
real ERCOT wind forecasts, prices, and load, so the modeling pipeline (OLS,
LightGBM, quantile regression, walk-forward backtest) can be built and
tested end to end before real ERCOT data is wired in.

Every file this script writes gets "SYNTHETIC" in the filename and a
synthetic_data_flag column, on purpose, so there's no way to accidentally
present this as real in a notebook, a chart, or a resume bullet.

Ground truth relationship we're building in (so we know what the models
*should* recover):
  - DART_spread(h) = DA_price(h) - RT_price(h)
  - forecast_error(h) = actual_wind(h) - forecast_wind(h)
  - True relationship: DART_spread = f(forecast_error) + noise
    where f is MOSTLY linear but has a nonlinear kink: large positive
    forecast errors (actual >> forecast, oversupply) cause a
    disproportionately large negative spread (curtailment-like dynamics),
    which is exactly the kind of nonlinearity that justifies using
    LightGBM over plain OLS.
  - Net load, hour-of-day, and season are added as confounders/controls
    that also move the spread, independent of forecast error.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)  # reproducible — document this in the README

N_DAYS = 400  # ~13 months hourly, gives walk-forward enough history
START = datetime(2024, 1, 1)
HOURS = N_DAYS * 24

timestamps = [START + timedelta(hours=h) for h in range(HOURS)]
df = pd.DataFrame({"timestamp": timestamps})
df["hour"] = df["timestamp"].dt.hour
df["doy"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month

# --- Wind capacity / forecast / actual ---
# Seasonal wind capacity factor: higher in spring, lower in summer (roughly
# matches real ERCOT seasonal wind patterns, simplified)
seasonal_wind_cf = 0.45 + 0.15 * np.sin(2 * np.pi * (df["doy"] - 80) / 365)
diurnal_wind_cf = 0.05 * np.sin(2 * np.pi * (df["hour"] - 3) / 24)  # nighttime-peaking wind
wind_capacity_mw = 30000  # rough ERCOT-scale installed wind capacity

true_wind_cf = np.clip(
    seasonal_wind_cf + diurnal_wind_cf + np.random.normal(0, 0.08, HOURS),
    0.02, 0.95,
)
actual_wind_mw = true_wind_cf * wind_capacity_mw

# Forecast = actual + forecast error. Forecast error has its own structure:
# mostly small/random, but fatter-tailed during certain weather regimes
# (simulated as a mixture: 85% small error, 15% larger "front moving
# through" error).
is_large_error_regime = np.random.binomial(1, 0.15, HOURS)
small_error = np.random.normal(0, 800, HOURS)       # MW
large_error = np.random.normal(0, 3500, HOURS)      # MW, fatter tail
forecast_error_mw = np.where(is_large_error_regime, large_error, small_error)

forecast_wind_mw = np.clip(actual_wind_mw - forecast_error_mw, 0, wind_capacity_mw)
# Recompute realized error from the clipped forecast so it's internally consistent
forecast_error_mw = actual_wind_mw - forecast_wind_mw

df["actual_wind_mw"] = actual_wind_mw
df["forecast_wind_mw"] = forecast_wind_mw
df["forecast_error_mw"] = forecast_error_mw

# --- Load (demand), net load ---
seasonal_load = 45000 + 15000 * np.sin(2 * np.pi * (df["doy"] - 200) / 365) ** 2  # summer peak
diurnal_load = 8000 * np.sin(2 * np.pi * (df["hour"] - 15) / 24) ** 2
load_mw = seasonal_load + diurnal_load + np.random.normal(0, 1500, HOURS)
df["load_mw"] = load_mw
df["net_load_mw"] = df["load_mw"] - df["actual_wind_mw"]

# --- DART spread: the actual target ---
# True relationship (this is what the models are supposed to recover):
#   - Linear component: beta * forecast_error
#   - Nonlinear kink: large oversupply errors (forecast_error very positive,
#     meaning way more wind than expected) cause a much bigger negative
#     price impact than the linear term alone predicts — this simulates
#     curtailment / negative-price dynamics during oversupply.
#   - Net load effect: tighter grid (high net load) amplifies the price
#     impact of any given forecast miss.
#   - Noise: real markets are noisy; don't make this look too clean.

linear_component = -0.004 * df["forecast_error_mw"]

# Kink: smooth (not a hard on/off switch — real grids don't have one),
# scaled by a softplus-like curve so it's negligible for small/moderate
# errors and grows disproportionately for large oversupply errors. This
# is the nonlinearity OLS can't capture but LightGBM should pick up.
KINK_THRESHOLD = 1200  # MW, roughly the 85th percentile of forecast_error_mw
EXCESS_CAP = 9000  # MW — caps the kink so fat-tail forecast misses don't
                    # produce implausible hundreds-of-dollars spreads
excess = np.clip(df["forecast_error_mw"] - KINK_THRESHOLD, 0, EXCESS_CAP)
kink_component = -0.00015 * excess ** 1.5

net_load_interaction = -0.00002 * df["forecast_error_mw"] * (50000 - df["net_load_mw"]) / 1000

# Regime label for later analysis (top 10% of forecast-error hours) —
# this is a LABEL for slicing results, not the functional form itself.
oversupply_flag = df["forecast_error_mw"] > df["forecast_error_mw"].quantile(0.90)

noise = np.random.normal(0, 4.5, HOURS)  # $/MWh noise, deliberately not tiny

df["dart_spread"] = (
    linear_component + kink_component + net_load_interaction + noise
)
# Explicit, documented floor/ceiling: even in the fattest-tail synthetic
# scenario, don't let the spread exceed what's plausible for real ERCOT
# oversupply/curtailment events. This is a deliberate modeling choice,
# not a silent clip — it matters for any tail-risk claim made later.
df["dart_spread"] = np.clip(df["dart_spread"], -150, 80)

# Back out DA and RT prices from the spread + a plausible RT price level
# (not meant to be realistic price levels, just internally consistent)
base_price = 28 + 12 * np.sin(2 * np.pi * (df["doy"] - 200) / 365) ** 2
rt_noise = np.random.normal(0, 6, HOURS)
df["rt_price"] = np.clip(base_price + rt_noise, -20, 400)
df["da_price"] = df["rt_price"] + df["dart_spread"]

# --- Flags ---
df["synthetic_data_flag"] = True
df["oversupply_regime_flag"] = oversupply_flag

OUT_PATH = "/home/claude/ercot_project/data/SYNTHETIC_ercot_hourly.csv"
df.to_csv(OUT_PATH, index=False)

print(f"Wrote {len(df)} rows to {OUT_PATH}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nColumn summary:")
print(df[[
    "actual_wind_mw", "forecast_wind_mw", "forecast_error_mw",
    "net_load_mw", "dart_spread", "da_price", "rt_price",
]].describe())
print(f"\nOversupply regime hours: {oversupply_flag.sum()} ({oversupply_flag.mean()*100:.1f}%)")
