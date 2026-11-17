"""
Synthetic stand-in for ERCOT hourly data (wind forecast/actual, load, prices).

NOT REAL DATA. I used this to build and test the modeling pipeline before I
had real ERCOT data. Output filenames contain SYNTHETIC and every row has
synthetic_data_flag = True.

Relationship built in, so I know what the models should recover:
  DART_spread = DA_price - RT_price
  forecast_error = actual_wind - forecast_wind
  DART_spread = f(forecast_error) + net-load interaction + noise, where f is
  mostly linear with a kink: large positive errors (oversupply) push the
  spread down disproportionately, roughly like curtailment.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)

N_DAYS = 400
START = datetime(2024, 1, 1)
HOURS = N_DAYS * 24

timestamps = [START + timedelta(hours=h) for h in range(HOURS)]
df = pd.DataFrame({"timestamp": timestamps})
df["hour"] = df["timestamp"].dt.hour
df["doy"] = df["timestamp"].dt.dayofyear
df["month"] = df["timestamp"].dt.month

# Wind: higher in spring, lower in summer, peaks at night. Rough ERCOT shape.
seasonal_wind_cf = 0.45 + 0.15 * np.sin(2 * np.pi * (df["doy"] - 80) / 365)
diurnal_wind_cf = 0.05 * np.sin(2 * np.pi * (df["hour"] - 3) / 24)
wind_capacity_mw = 30000

true_wind_cf = np.clip(
    seasonal_wind_cf + diurnal_wind_cf + np.random.normal(0, 0.08, HOURS),
    0.02, 0.95,
)
actual_wind_mw = true_wind_cf * wind_capacity_mw

# Forecast error is a mixture: 85% small misses, 15% big ones (a front
# coming through faster or slower than forecast).
is_large_error_regime = np.random.binomial(1, 0.15, HOURS)
small_error = np.random.normal(0, 800, HOURS)
large_error = np.random.normal(0, 3500, HOURS)
forecast_error_mw = np.where(is_large_error_regime, large_error, small_error)

forecast_wind_mw = np.clip(actual_wind_mw - forecast_error_mw, 0, wind_capacity_mw)
# recompute after clipping so error = actual - forecast exactly
forecast_error_mw = actual_wind_mw - forecast_wind_mw

df["actual_wind_mw"] = actual_wind_mw
df["forecast_wind_mw"] = forecast_wind_mw
df["forecast_error_mw"] = forecast_error_mw

seasonal_load = 45000 + 15000 * np.sin(2 * np.pi * (df["doy"] - 200) / 365) ** 2  # summer peak
diurnal_load = 8000 * np.sin(2 * np.pi * (df["hour"] - 15) / 24) ** 2
load_mw = seasonal_load + diurnal_load + np.random.normal(0, 1500, HOURS)
df["load_mw"] = load_mw
df["net_load_mw"] = df["load_mw"] - df["actual_wind_mw"]

linear_component = -0.004 * df["forecast_error_mw"]

# Smooth kink above ~the 85th percentile of forecast error, capped so the
# fat-tail misses don't produce absurd spreads.
KINK_THRESHOLD = 1200
EXCESS_CAP = 9000
excess = np.clip(df["forecast_error_mw"] - KINK_THRESHOLD, 0, EXCESS_CAP)
kink_component = -0.00015 * excess ** 1.5

# tighter grid (higher net load) amplifies the effect of a miss
net_load_interaction = -0.00002 * df["forecast_error_mw"] * (50000 - df["net_load_mw"]) / 1000

# label only, used for slicing results later
oversupply_flag = df["forecast_error_mw"] > df["forecast_error_mw"].quantile(0.90)

noise = np.random.normal(0, 4.5, HOURS)

df["dart_spread"] = (
    linear_component + kink_component + net_load_interaction + noise
)
df["dart_spread"] = np.clip(df["dart_spread"], -150, 80)

# DA and RT prices backed out from the spread, only meant to be consistent
base_price = 28 + 12 * np.sin(2 * np.pi * (df["doy"] - 200) / 365) ** 2
rt_noise = np.random.normal(0, 6, HOURS)
df["rt_price"] = np.clip(base_price + rt_noise, -20, 400)
df["da_price"] = df["rt_price"] + df["dart_spread"]

df["synthetic_data_flag"] = True
df["oversupply_regime_flag"] = oversupply_flag

OUT_PATH = os.path.join(ROOT, "data", "SYNTHETIC_ercot_hourly.csv")
df.to_csv(OUT_PATH, index=False)

print(f"Wrote {len(df)} rows to {OUT_PATH}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nColumn summary:")
print(df[[
    "actual_wind_mw", "forecast_wind_mw", "forecast_error_mw",
    "net_load_mw", "dart_spread", "da_price", "rt_price",
]].describe())
print(f"\nOversupply regime hours: {oversupply_flag.sum()} ({oversupply_flag.mean()*100:.1f}%)")
