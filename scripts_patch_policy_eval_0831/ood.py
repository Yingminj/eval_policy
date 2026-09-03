#!/usr/bin/env python
"""How far outside the training action range do the held-out EEF episodes sit?

The checkpoint normalises with MIN_MAX statistics computed on ``batch_success_505_eef``.
Anything outside that box in the held-out episodes is a value the normaliser has never
had to represent -- and, for the *ground truth*, a target the policy cannot emit at all
once its own output is un-normalised from [-1, 1].
"""
import glob
import json

import numpy as np
import pandas as pd

D = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"
st = json.load(open(D + "batch_success_505_eef/meta/stats.json"))["action"]
lo, hi = np.array(st["min"], np.float32), np.array(st["max"], np.float32)
names = json.load(open(D + "batch_success_533_eef/meta/info.json"))["features"]["action"]["names"]
keep = set(json.load(open("splits.json"))["unseen_by_505_eef"])

files = sorted(glob.glob(D + "batch_success_533_eef/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
df = df[df["episode_index"].isin(keep)]
a = np.stack(df["action"].to_numpy()).astype(np.float32)
out = (a < lo) | (a > hi)
print(f"held-out frames: {len(a)}")
print(f"frames with at least one out-of-range channel: {out.any(1).sum()} ({100 * out.any(1).mean():.2f}%)")
print("\n| channel | out-of-range frames | worst overshoot (raw units) |")
print("|---|---:|---:|")
for i, n in enumerate(names):
    over = np.maximum(a[:, i] - hi[i], lo[i] - a[:, i]).max()
    print(f"| {n} | {out[:, i].sum()} ({100 * out[:, i].mean():.2f}%) | {max(over, 0):.4f} |")
