#!/usr/bin/env python
"""How far outside the training action range does the independent 53-episode EEF eval set sit?

Both checkpoints normalise with MIN_MAX statistics computed on ``batch_success_505_eef``.
Anything outside that box is a value the normaliser has never had to represent -- and, for
the *ground truth*, a target the policy cannot emit at all once its own output is
un-normalised from [-1, 1].  Reported for the new independent eval set and, for contrast,
for the random-interleave 67-episode held-out split the 08-31 report had to use.
"""
import glob
import json

import numpy as np
import pandas as pd

D = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"
st = json.load(open(D + "batch_success_505_eef/meta/stats.json"))["action"]
lo, hi = np.array(st["min"], np.float32), np.array(st["max"], np.float32)
names = json.load(open(D + "batch_success_505_eef/meta/info.json"))["features"]["action"]["names"]


def load(name, keep=None):
    files = sorted(glob.glob(D + name + "/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    if keep is not None:
        df = df[df["episode_index"].isin(keep)]
    return np.stack(df["action"].to_numpy()).astype(np.float32)


old = json.load(open("../scripts_patch_policy_eval_0831/splits.json"))["unseen_by_505_eef"]
for label, a in [("53-episode independent eval set", load("batch_success_53_eval_data_eef")),
                 ("67-episode random held-out (08-31)", load("batch_success_533_eef", set(old)))]:
    out = (a < lo) | (a > hi)
    print(f"\n## {label}")
    print(f"frames: {len(a)}")
    print(f"frames with at least one out-of-range channel: {out.any(1).sum()} ({100 * out.any(1).mean():.2f}%)")
    print("\n| channel | out-of-range frames | worst overshoot (raw units) |")
    print("|---|---:|---:|")
    for i, n in enumerate(names):
        over = np.maximum(a[:, i] - hi[i], lo[i] - a[:, i]).max()
        print(f"| {n} | {out[:, i].sum()} ({100 * out[:, i].mean():.2f}%) | {max(over, 0):.4f} |")
