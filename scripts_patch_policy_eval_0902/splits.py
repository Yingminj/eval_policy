#!/usr/bin/env python
"""Is the new EEF eval set the same 53 recordings as the joint one, and unseen by every train set?

Same two fingerprints as ../scripts_patch_policy_eval_0831/splits.py:
  * ``observation.velocity`` -- 16-D in both action spaces, so it identifies a recording
    across the joint/EEF re-projection.
  * ``action``               -- the key ``offline_chunk_eval.py`` filters on.

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


EVAL = ["batch_success_53_eval_data", "batch_success_53_eval_data_eef"]
TRAIN = ["batch_success_361", "batch_success_505",
         "batch_success_361_eef", "batch_success_505_eef", "batch_success_533_eef"]

vel = {n: fingerprint(n, "observation.velocity") for n in EVAL + TRAIN}
act = {n: fingerprint(n, "action") for n in EVAL + TRAIN}

print("## episode counts\n")
for n in EVAL + TRAIN:
    print(f"{n:32s} {len(vel[n]):4d} episodes")

print("\n## is the EEF eval set the same recordings as the joint eval set?\n")
j, e = vel["batch_success_53_eval_data"], vel["batch_success_53_eval_data_eef"]
shared = set(j.values()) & set(e.values())
print(f"observation.velocity fingerprints shared: {len(shared)} / {len(j)}")
inv = {h: ep for ep, h in j.items()}
mismatch = sorted(ep for ep, h in e.items() if h not in inv)
print(f"EEF episodes with no joint counterpart: {mismatch}")
remap = {ep: inv[h] for ep, h in e.items() if h in inv}
print(f"episode-index permutation identity: {all(k == v for k, v in remap.items())}")

print("\n## contamination: eval episodes present in any training set\n")
for n in TRAIN:
    hv, ha = set(vel[n].values()), set(act[n].values())
    for ev in EVAL:
        nv = len(set(vel[ev].values()) & hv)
        na = len(set(act[ev].values()) & ha)
        print(f"{ev:32s} in {n:24s}: velocity {nv:3d}  action {na:3d}")

json.dump({"velocity_shared": len(shared), "eef_to_joint_episode_map": remap},
          open("splits.json", "w"), indent=1)
