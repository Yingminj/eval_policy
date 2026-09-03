#!/usr/bin/env python
"""Was the new EEF eval set produced by the same transform as the EEF training set?

The 14-D EEF datasets are not re-conversions of the rosbags: `tool/tr_joint_to_eef.py`
runs forward kinematics on the joint dataset's own `observation.state`/`action` with the
M6-696 MJCF.  The eval set's `meta/conversion_manifest.json` is still the *joint*
conversion's manifest (it is a hard link to the joint dataset's copy), so there is no
recorded provenance -- it has to be checked against the data.

This re-runs that FK on the joint datasets and compares with the shipped EEF datasets.
If both reproduce, the eval set went through exactly the transform the training set did,
and the comparison is apples to apples.

Run: /opt/robot-platform/train-venv/bin/python fk_provenance.py > fk_provenance.txt
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/kewei/YING/robot_data_platform/tool")
from tr_joint_to_eef import MjcfForwardKinematics  # noqa: E402

D = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"
MJCF = "/home/kewei/YING/Apex_Deploy_new/robot_node/marvin_description/mjcf/matrix/m6_696.xml"
GRIPPERS = ["gripper_L", "gripper_R"]
N = 4000  # frames sampled per dataset; FK is exact, so a sample settles it


def cols(name, keys):
    files = sorted(glob.glob(f"{D}{name}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(f, columns=list(keys)) for f in files])
    return {k: np.stack(df[k].to_numpy()) for k in keys}


for joint, eef in [("batch_success_505", "batch_success_505_eef"),
                   ("batch_success_53_eval_data", "batch_success_53_eval_data_eef")]:
    jinfo = json.load(open(f"{D}{joint}/meta/info.json"))
    j = cols(joint, ["observation.state", "action"])
    e = cols(eef, ["observation.state", "action"])
    print(f"\n## {joint} -> {eef}")
    for key in ("observation.state", "action"):
        names = jinfo["features"][key]["names"]
        fk = MjcfForwardKinematics(MJCF, names, "left_tool", "right_tool", "euler")
        gi = [names.index(g) for g in GRIPPERS]
        sel = np.linspace(0, len(j[key]) - 1, N).astype(int)
        src = j[key][sel].astype(np.float64)
        left, right = fk.evaluate(src)
        mine = np.concatenate([left, right, src[:, gi].astype(np.float32)], axis=1)
        theirs = e[key][sel].astype(np.float32)
        d = np.abs(mine - theirs)
        print(f"  {key:18s} shape {theirs.shape[1]}D  max|Δ| = {d.max():.3e}  "
              f"mean|Δ| = {d.mean():.3e}  {'MATCH' if d.max() < 1e-4 else 'MISMATCH'}")
        if d.max() >= 1e-4:
            print("   worst channel:", int(d.max(0).argmax()), "→", d.max(0).round(5).tolist())
