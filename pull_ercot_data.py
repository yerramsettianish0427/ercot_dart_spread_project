"""
Pull ERCOT public data with gridstatus into data/real_ercot_raw/.

Run this on a normal machine. ERCOT blocks many cloud/datacenter IPs.

    pip install gridstatus pandas
    python pull_ercot_data.py

Writes wind_actual_forecast.csv, solar_actual_forecast.csv, dam_spp.csv,
rtm_spp.csv and load_weather_zone.csv. Asks for the last 90 days, but the
public RTM report only keeps about 7 days.
"""

import os
from datetime import datetime, timedelta

import gridstatus
from gridstatus import Markets

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_ercot_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=90)

print(f"Pulling ERCOT data from {START_DATE} to {END_DATE}")

iso = gridstatus.Ercot()


def safe_pull(label, fn, *args, **kwargs):
    """One failed report shouldn't stop the rest."""
    print(f"\n--- {label} ---")
    try:
        df = fn(*args, **kwargs)
        print(f"  OK: {df.shape[0]} rows, {df.shape[1]} cols")
        print(f"  Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        return None


wind = safe_pull(
    "Wind actual + forecast (hourly)",
    iso.get_wind_actual_and_forecast_hourly,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if wind is not None:
    wind.to_csv(f"{OUTPUT_DIR}/wind_actual_forecast.csv", index=False)

solar = safe_pull(
    "Solar actual + forecast (hourly)",
    iso.get_solar_actual_and_forecast_hourly,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if solar is not None:
    solar.to_csv(f"{OUTPUT_DIR}/solar_actual_forecast.csv", index=False)

# DAM SPP over 90 days can be slow; shrink the range if it times out
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
    print(f"DAM pull failed: {e}")

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
    print(f"RTM pull failed: {e}")

load = safe_pull(
    "Load by weather zone",
    iso.get_load_by_weather_zone,
    date=START_DATE.isoformat(),
    end=END_DATE.isoformat(),
)
if load is not None:
    load.to_csv(f"{OUTPUT_DIR}/load_weather_zone.csv", index=False)

print(f"\nDone. CSVs are in {OUTPUT_DIR}")
