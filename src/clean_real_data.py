"""
Clean and join the real ERCOT pulls (gridstatus, pulled 2026-10-11) into one
hourly table at HB_HOUSTON.

What the raw files look like:
  - The wind/solar report is republished every hour with a 48h-back /
    168h-forward window, so each Interval Start has many rows. GEN (actual)
    is null until the hour settles, then stays constant. STWPF (forecast)
    keeps changing across publishes.
  - Only 234 of 383 wind hours have settled; the rest are future forecasts
    and get dropped.
  - Public RTM prices only go back 7 days, so the final overlap is about a
    week (~166 hours).

Per hour I keep the last settled GEN as the actual, and the STWPF from the
last publish strictly before the hour started as the forecast. Using any
later forecast would leak.
"""

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "real_ercot_raw")
OUT = os.path.join(ROOT, "data", "REAL_ercot_hourly.csv")

HUB = "HB_HOUSTON"

wind = pd.read_csv(f"{RAW}/wind_actual_forecast.csv", parse_dates=["Interval Start", "Publish Time"])
wind = wind.sort_values(["Interval Start", "Publish Time"])

actual = (
    wind.dropna(subset=["GEN SYSTEM WIDE"])
    .groupby("Interval Start")
    .last()["GEN SYSTEM WIDE"]
    .rename("actual_wind_mw")
)


def last_forecast_before_settlement(group):
    interval_start = group.name
    prior = group[group["Publish Time"] < interval_start]
    if len(prior) == 0:
        return np.nan
    return prior.sort_values("Publish Time").iloc[-1]["STWPF SYSTEM WIDE"]


forecast = wind.groupby("Interval Start").apply(last_forecast_before_settlement, include_groups=False)
forecast = forecast.rename("forecast_wind_mw")

wind_clean = pd.concat([actual, forecast], axis=1).reset_index()
wind_clean = wind_clean.dropna(subset=["actual_wind_mw", "forecast_wind_mw"])
wind_clean["forecast_error_mw"] = wind_clean["actual_wind_mw"] - wind_clean["forecast_wind_mw"]

print(f"Wind: {len(wind_clean)} hours with a settled actual and a "
      f"pre-interval forecast")

# Tradeable feature: how much the forecast moved between 6h and 1h before
# the interval. Both snapshots exist before the hour starts, so unlike
# forecast_error_mw this doesn't need the actual.
wind_sorted = wind.sort_values("Publish Time")
targets = wind_clean[["Interval Start"]].copy()
targets["t_minus_6h"] = targets["Interval Start"] - pd.Timedelta(hours=6)
targets["t_minus_1h"] = targets["Interval Start"] - pd.Timedelta(hours=1)

snap_6h = pd.merge_asof(
    targets[["t_minus_6h"]].rename(columns={"t_minus_6h": "asof"}).sort_values("asof"),
    wind_sorted[["Publish Time", "STWPF SYSTEM WIDE"]].rename(columns={"Publish Time": "asof"}),
    on="asof", direction="backward",
).rename(columns={"STWPF SYSTEM WIDE": "fcst_t_minus_6h"})

snap_1h = pd.merge_asof(
    targets[["t_minus_1h"]].rename(columns={"t_minus_1h": "asof"}).sort_values("asof"),
    wind_sorted[["Publish Time", "STWPF SYSTEM WIDE"]].rename(columns={"Publish Time": "asof"}),
    on="asof", direction="backward",
).rename(columns={"STWPF SYSTEM WIDE": "fcst_t_minus_1h"})

# merge_asof output is sorted by asof, so line targets up the same way
targets = targets.sort_values("t_minus_6h")
targets["fcst_t_minus_6h"] = snap_6h["fcst_t_minus_6h"].values
targets = targets.sort_values("t_minus_1h")
targets["fcst_t_minus_1h"] = snap_1h["fcst_t_minus_1h"].values

targets["forecast_revision_mw"] = targets["fcst_t_minus_1h"] - targets["fcst_t_minus_6h"]
wind_clean = wind_clean.merge(
    targets[["Interval Start", "forecast_revision_mw"]], on="Interval Start", how="left"
)
print(f"Forecast revision feature built for "
      f"{wind_clean['forecast_revision_mw'].notna().sum()}/{len(wind_clean)} hours")

dam = pd.read_csv(f"{RAW}/dam_spp.csv", parse_dates=["Time"])
dam_hub = dam[dam["Location"] == HUB][["Time", "SPP"]].rename(
    columns={"Time": "Interval Start", "SPP": "da_price"}
)

# RTM is 15-minute, average to hourly
rtm = pd.read_csv(f"{RAW}/rtm_spp.csv", parse_dates=["Time"])
rtm_hub = rtm[rtm["Location"] == HUB].copy()
rtm_hub["Interval Start"] = rtm_hub["Time"].dt.floor("h")
rtm_hourly = rtm_hub.groupby("Interval Start")["SPP"].mean().rename("rt_price").reset_index()

# total load, not net load (this file has no renewables subtracted)
load = pd.read_csv(f"{RAW}/load_weather_zone.csv", parse_dates=["Time"])
load_clean = load[["Time", "System Total"]].rename(
    columns={"Time": "Interval Start", "System Total": "load_mw"}
)

df = wind_clean.merge(dam_hub, on="Interval Start", how="inner")
df = df.merge(rtm_hourly, on="Interval Start", how="inner")
df = df.merge(load_clean, on="Interval Start", how="inner")

df["dart_spread"] = df["da_price"] - df["rt_price"]
df["hour"] = df["Interval Start"].dt.hour
df["doy"] = df["Interval Start"].dt.dayofyear
df = df.sort_values("Interval Start").reset_index(drop=True)

print(f"\n=== Final merged real dataset ===")
print(f"Rows: {len(df)}")
print(f"Date range: {df['Interval Start'].min()} to {df['Interval Start'].max()}")
print(f"Hub: {HUB}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\n{df[['actual_wind_mw','forecast_wind_mw','forecast_error_mw','da_price','rt_price','dart_spread','load_mw']].describe()}")

if len(df) < 200:
    print(f"\nOnly {len(df)} hours, treat any result as a pilot.")

df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
