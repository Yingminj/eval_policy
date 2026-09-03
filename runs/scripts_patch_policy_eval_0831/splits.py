#!/usr/bin/env python
"""Which episodes may each checkpoint legitimately be scored on?

The dataset names in this corpus are cumulative merges, so containment has to be decided
on the data. Two fingerprints are used:

  * ``action``            -- the same key ``offline_chunk_eval.py`` filters on.
  * ``observation.velocity`` -- 16-D in *both* the joint and the EEF datasets, so it is the
    only key that can match an episode across the two action spaces.  That is what proves
    ``batch_success_505_eef`` is a re-projection of ``batch_success_505`` rather than a
    separate recording, and that the 53-episode eval set appears in neither.

Run: /opt/robot-platform/train-venv/bin/python splits.py > splits.txt
"""
import glob
import hashlib
import json

import numpy as np
import pandas as pd

D = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"


def fingerprint(name: str, column: str) -> dict[int, str]:
    files = sorted(glob.glob(f"{D}{name}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(name)
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", column]) for f in files])
    return {
        int(ep): hashlib.sha1(np.stack(g[column].to_numpy()).astype(np.float32).tobytes()).hexdigest()
        for ep, g in df.groupby("episode_index")
    }


JOINT = ["batch_success_53_eval_data", "batch_success_361", "batch_success_505"]
EEF = ["batch_success_361_eef", "batch_success_505_eef", "batch_success_533_eef"]

vel = {n: fingerprint(n, "observation.velocity") for n in JOINT + EEF}
act = {n: fingerprint(n, "action") for n in JOINT + EEF}

print("## episode counts\n")
for n in JOINT + EEF:
    print(f"{n:32s} {len(vel[n]):4d} episodes")

print("\n## cross-space identity (observation.velocity)\n")
h505, h505e = set(vel["batch_success_505"].values()), set(vel["batch_success_505_eef"].values())
print(f"batch_success_505 vs _eef, shared episodes: {len(h505 & h505e)} / {len(h505)}")
ev = set(vel["batch_success_53_eval_data"].values())
for n in JOINT[1:] + EEF:
    print(f"53-episode eval set inside {n:26s}: {len(ev & set(vel[n].values()))}")

print("\n## EEF held-out split (action fingerprint, the key the harness filters on)\n")
a533, a505, a361 = act["batch_success_533_eef"], act["batch_success_505_eef"], act["batch_success_361_eef"]
h505a, h361a = set(a505.values()), set(a361.values())
out505 = sorted(ep for ep, h in a533.items() if h not in h505a)
out_both = sorted(ep for ep, h in a533.items() if h not in h505a and h not in h361a)
print(f"533_eef episodes unseen by patch_policy(505_eef):        {len(out505)}")
print(f"533_eef episodes unseen by BOTH it and act_eef(361_eef): {len(out_both)}")
print(f"unseen-by-505 indices: {out505}")
print(f"unseen-by-both indices: {out_both}")
json.dump({"unseen_by_505_eef": out505, "unseen_by_both": out_both}, open("splits.json", "w"), indent=1)
