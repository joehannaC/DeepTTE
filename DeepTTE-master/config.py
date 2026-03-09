import json
import numpy as np

dist = []
time = []
lngs = []
lats = []
time_gap = []
dist_gap = []

with open("/Users/joehannacansino/Desktop/THS3/DeepTTE-master/data/train_data.jsonl") as f:
    for line in f:
        d = json.loads(line)

        dist.append(d["dist"])
        time.append(d["time"])

        lngs.extend(d["lngs"])
        lats.extend(d["lats"])
        time_gap.extend(d["time_gap"])
        dist_gap.extend(d["dist_gap"])

config = {
    "dist_gap_mean": np.mean(dist_gap),
    "dist_gap_std": np.std(dist_gap),
    "time_gap_mean": np.mean(time_gap),
    "time_gap_std": np.std(time_gap),
    "lngs_mean": np.mean(lngs),
    "lngs_std": np.std(lngs),
    "lats_mean": np.mean(lats),
    "lats_std": np.std(lats),
    "dist_mean": np.mean(dist),
    "dist_std": np.std(dist),
    "time_mean": np.mean(time),
    "time_std": np.std(time)
}

print(config)