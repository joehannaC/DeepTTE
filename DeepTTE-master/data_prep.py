import pandas as pd
import json
import numpy as np

df = pd.read_csv("/Users/joehannacansino/Desktop/THS3/data/train_data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

# Sort properly
df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

samples = []

for trip_id, g in df.groupby("trip_id"):
    g = g.sort_values("timestamp").copy()

    # Keep only valid coordinate rows
    g = g[g["latitude"].notna() & g["longitude"].notna()].copy()
    if len(g) == 0:
        continue

    # Basic IDs
    driver_id = int(g["vehicleID"].iloc[0]) if pd.notna(g["vehicleID"].iloc[0]) else -1
    service_date = g["service_date"].iloc[0]
    start_ts = g["timestamp"].iloc[0]

    date_id = int(service_date.day - 1)      # 0 to 30
    week_id = int(service_date.weekday())    # Mon=0 ... Sun=6
    time_id = int(start_ts.hour * 60 + start_ts.minute)

    # Trip-level unique segment totals
    # One total time + one total distance per segment only
    seg_totals = (
        g.dropna(subset=["segment"])
         .sort_values(["segment", "timestamp"])
         .groupby("segment", as_index=False)[["total_travel_time_sec", "total_distance_m"]]
         .first()
    )

    time_min = float(
        seg_totals["total_travel_time_sec"].fillna(0).clip(lower=0).sum()
    ) / 60.0

    dist_km = float(
        seg_totals["total_distance_m"].fillna(0).clip(lower=0).sum()
    ) / 1000.0

    # Trajectory sequences

    lngs = g["longitude"].astype(float).tolist()
    lats = g["latitude"].astype(float).tolist()

    # Use currentStatus as states
    states = g["currentStatus"].fillna(0).astype(int).tolist()

    # time_gap: elapsed time from first point
    time_gap = (
        g["travel_time_sec"]
        .fillna(0)
        .clip(lower=0)
        .astype(float) / 60.0
    ).tolist()

    if len(time_gap) > 0:
        time_gap[0] = 0.0

    # dist_gap: cumulative row-level distance from first point
    dist_gap = (
        g["distance_m"]
        .fillna(0)
        .clip(lower=0)
        .astype(float)
        .cumsum() / 1000.0
    ).tolist()

    if len(dist_gap) > 0:
        dist_gap[0] = 0.0

    sample = {
        "driverID": driver_id,
        "tripID": str(trip_id),
        "dateID": date_id,
        "weekID": week_id,
        "timeID": time_id,
        "dist": dist_km,
        "time": time_min,
        "lngs": lngs,
        "lats": lats,
        "states": states,
        "time_gap": time_gap,
        "dist_gap": dist_gap
    }

    samples.append(sample)

# Write JSONL
with open("train_data.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")