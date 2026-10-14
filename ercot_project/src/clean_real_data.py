"""
Clean and merge REAL ERCOT data (pulled via gridstatus on 2026-10-11).

Key findings from inspecting the raw pulls that drive every decision below:
  1. wind_actual_forecast.csv / solar_actual_forecast.csv are a ROLLING
     report: each Interval Start has many rows, one per hourly re-publish,
     because ERCOT reposts a 48h-trailing + 168h-forward forecast every
     hour. GEN (actual) is null until the interval settles (~1-2h after
     Interval Start), then locks to a constant value across all later
     republishes.
  2. Only 234 of 383 unique Interval Start hours in the wind file have
     actually settled (GEN is non-null) — the rest are ERCOT's own future
     forecast for hours that haven't happened yet and MUST be excluded.
  3. RTM price history only goes back 7 days (public retention window),
     which — intersected with settled wind hours and DAM coverage — leaves
     roughly ONE WEEK of real, fully-overlapping, non-leaking data
     (~150-170 hours). This is the honest real-data sample size, nowhere
     near the 90 days used in the synthetic version, and every result
     downstream must be reported with that caveat loud and first.

Leakage-safe construction:
  - actual_wind_mw: the LAST (post-settlement) non-null GEN value per hour
    — this is only usable for the EXPLANATORY model, never a tradeable one.
  - forecast_wind_mw: the STWPF value from the LAST publish snapshot that
    is still strictly BEFORE the interval started — i.e. the most recent
    forecast actually available before real-time, not an after-the-fact
    one. This is what keeps forecast_error_mw honest.
"""

import pandas as pd
import numpy as np

RAW = "/home/claude/ercot_project/data/real_ercot_raw"
OUT = "/home/claude/ercot_project/data/REAL_ercot_hourly.csv"

HUB = "HB_HOUSTON"

# --- Wind: dedupe to one settled-actual + one pre-settlement forecast per hour ---
wind = pd.read_csv(f"{RAW}/wind_actual_forecast.csv", parse_dates=["Interval Start", "Publish Time"])
wind = wind.sort_values(["Interval Start", "Publish Time"])

# Settled actual: last non-null GEN per Interval Start
actual = (
    wind.dropna(subset=["GEN SYSTEM WIDE"])
    .groupby("Interval Start")
    .last()["GEN SYSTEM WIDE"]
    .rename("actual_wind_mw")
)

# Pre-settlement forecast: for each Interval Start, the STWPF from the
# latest publish strictly before that hour started
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

print(f"Wind: {len(wind_clean)} hours with both a settled actual AND a "
      f"pre-settlement forecast (leakage-safe)")

# --- TRADEABLE feature: forecast REVISION, not actual. ---
# ERCOT republishes STWPF roughly hourly. A desk can observe how the
# forecast has moved between two points *before* the interval starts and
# act on that momentum without ever seeing the actual. This is the
# leakage-free alternative to forecast_error_mw (which needs the actual,
# unknowable before settlement). Built via merge_asof: for each Interval
# Start, grab the latest STWPF published at/before (Interval Start - 6h)
# and at/before (Interval Start - 1h); the revision is the difference.
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

# merge_asof sorted by 'asof' then needs re-aligning back to original order
targets = targets.sort_values("t_minus_6h")
targets["fcst_t_minus_6h"] = snap_6h["fcst_t_minus_6h"].values
targets = targets.sort_values("t_minus_1h")
targets["fcst_t_minus_1h"] = snap_1h["fcst_t_minus_1h"].values

targets["forecast_revision_mw"] = targets["fcst_t_minus_1h"] - targets["fcst_t_minus_6h"]
wind_clean = wind_clean.merge(
    targets[["Interval Start", "forecast_revision_mw"]], on="Interval Start", how="left"
)
print(f"Forecast-revision (tradeable) feature built for "
      f"{wind_clean['forecast_revision_mw'].notna().sum()}/{len(wind_clean)} hours")

# --- DAM price (hourly already) ---
dam = pd.read_csv(f"{RAW}/dam_spp.csv", parse_dates=["Time"])
dam_hub = dam[dam["Location"] == HUB][["Time", "SPP"]].rename(
    columns={"Time": "Interval Start", "SPP": "da_price"}
)

# --- RTM price (15-min -> hourly mean) ---
rtm = pd.read_csv(f"{RAW}/rtm_spp.csv", parse_dates=["Time"])
rtm_hub = rtm[rtm["Location"] == HUB].copy()
rtm_hub["Interval Start"] = rtm_hub["Time"].dt.floor("h")
rtm_hourly = rtm_hub.groupby("Interval Start")["SPP"].mean().rename("rt_price").reset_index()

# --- Load (System Total, as a net-load proxy — no separate net-load calc
# since this file is total load, not load-minus-renewables) ---
load = pd.read_csv(f"{RAW}/load_weather_zone.csv", parse_dates=["Time"])
load_clean = load[["Time", "System Total"]].rename(
    columns={"Time": "Interval Start", "System Total": "load_mw"}
)

# --- Merge everything on Interval Start (inner join = only real overlap) ---
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
    print(f"\n*** SAMPLE SIZE WARNING: {len(df)} hours is small. ***")
    print("Results from this dataset should be reported as a PILOT / proof of")
    print("concept on real data, not a validated finding. Confidence intervals")
    print("will be wide. This is honest scope, not a flaw to hide.")

df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
