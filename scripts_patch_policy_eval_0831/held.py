#!/usr/bin/env python
"""How often is action[t] a bit-identical repeat of action[t-1]?

`check_alignment.py`'s negative control tripped on the EEF set, which is worth a number:
a held command means the recorded target pose was published slower than the 30 Hz grid,
so part of what the policy is asked to predict is a sample-and-hold artefact rather than
motion.  Compared here against the joint-space sets, which come from the same bags.
"""
import glob

import numpy as np
import pandas as pd

D = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"
for name in ("batch_success_361", "batch_success_53_eval_data",
             "batch_success_505_eef", "batch_success_533_eef"):
    files = sorted(glob.glob(f"{D}{name}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    held = tot = 0
    for _, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(np.float32)
        held += int((a[1:] == a[:-1]).all(1).sum())
        tot += len(a) - 1
    print(f"{name:28s} held(all channels identical to previous frame): "
          f"{held}/{tot} = {100 * held / tot:.2f}%")
