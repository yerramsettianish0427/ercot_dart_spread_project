"""
Run this LOCALLY (not in a sandboxed/cloud environment) — ERCOT's site
blocks a lot of cloud/datacenter IP ranges and the gridstatus library
scrapes their public reports directly.

Setup:
    pip install gridstatus pandas

Usage:
    python pull_ercot_data.py

Output (in ./ercot_data/):
    wind_actual_forecast_houston.csv   -- hourly actual wind output vs STWPF forecast
    solar_actual_forecast_houston.csv  -- hourly actual solar output vs forecast
    dam_spp_houston.csv                -- day-ahead settlement point prices
    rtm_spp_houston.csv                -- real-time settlement point prices
    load_houston.csv                   -- hourly load by weather zone (control variable)

Date range: last 90 days by default (edit START_DATE below for more history —
more history = better model, but each extra month adds pull time).
"""

import gridstatus
from gridstatus import Markets
import pandas as pd
from datetime import datetime, timedelta
import os

OUTPUT_DIR = "ercot_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Date range ---
# Start conservative (90 days). You can re-run with an earlier START_DATE
# later to extend history once we confirm the pipeline works end-to-end.
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=90)

print(f"Pulling ERCOT data from {START_DATE} to {END_DATE}")
print("This will take a while — each report is pulled day by day or in chunks.")

iso = gridstatus.Ercot()

def safe_pull(label, fn, *args, **kwargs):
    """Wrap each pull so one failure doesn't kill the whole script."""
    print(f"\n--- Pulling: {label} ---")
    try:
        df = fn(*args, **kwargs)
        print(f"  OK: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None

# 1. Wind actual vs forecast (hourly, system-wide)
wind = safe_pull(
    "Wind actual + forecast (hourly)",
    iso.get_wind_actual_and_forecast_hourly,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if wind is not None:
    wind.to_csv(f"{OUTPUT_DIR}/wind_actual_forecast.csv", index=False)

# 2. Solar actual vs forecast (hourly, system-wide)
solar = safe_pull(
    "Solar actual + forecast (hourly)",
    iso.get_solar_actual_and_forecast_hourly,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if solar is not None:
    solar.to_csv(f"{OUTPUT_DIR}/solar_actual_forecast.csv", index=False)

# 3. Day-ahead settlement point prices (Houston hub)
#    Note: ERCOT DAM prices are typically pulled per-day; gridstatus handles
#    chunking internally for most methods, but DAM SPP can be slow over 90 days.
#    If this times out, reduce the date range and re-run.
try:
    dam = safe_pull(
        "Day-ahead settlement point prices",
        iso.get_spp,
        date=START_DATE.isoformat(),
        end=END_DATE.isoformat(),
        market=Markets.DAY_AHEAD_HOURLY,
        location_type="Trading Hub",
    )
    if dam is not None:
        dam.to_csv(f"{OUTPUT_DIR}/dam_spp.csv", index=False)
except Exception as e:
    print(f"DAM pull failed outright: {e}")

# 4. Real-time settlement point prices (Houston hub)
try:
    rtm = safe_pull(
        "Real-time settlement point prices",
        iso.get_spp,
        date=START_DATE.isoformat(),
        end=END_DATE.isoformat(),
        market=Markets.REAL_TIME_15_MIN,
        location_type="Trading Hub",
    )
    if rtm is not None:
        rtm.to_csv(f"{OUTPUT_DIR}/rtm_spp.csv", index=False)
except Exception as e:
    print(f"RTM pull failed outright: {e}")

# 5. Load by weather zone (control variable — net load matters more than
#    gross load, but load alone is a reasonable proxy to start with)
load = safe_pull(
    "Load by weather zone",
    iso.get_load_by_weather_zone,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if load is not None:
    load.to_csv(f"{OUTPUT_DIR}/load_weather_zone.csv", index=False)

print("\n\n=== DONE ===")
print(f"Check the '{OUTPUT_DIR}/' folder for CSVs.")
print("Upload whichever ones succeeded back to Claude, plus the full console")
print("output above (especially any FAILED lines and the column lists) —")
print("ERCOT's method signatures change between gridstatus versions, so we")
print("need to see the real output before writing the merge/cleaning code.")
