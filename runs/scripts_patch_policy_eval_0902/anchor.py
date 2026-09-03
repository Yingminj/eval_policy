#!/usr/bin/env python
"""Does the emitted chunk start where the arm actually is?

The 08-31 report's mechanism claim (§1.2) is that the EEF action space fixed pose
anchoring: the first step of the chunk lands near the measured pose instead of jumping.
It was measured on the easy random-interleave split and only for `patch_policy`, whose
`probe_conditioning.py` cannot load an ACT checkpoint.  This re-measures it on the
independent 53-episode eval set, for any policy type:

    policy_mae        mean |chunk[0] - measured pose|
    demonstration_mae mean |ground truth action[0] - measured pose|   (the ceiling)
    ratio             policy / demonstration.  1.0 = anchored as well as a human teleop
                      command; large = the chunk opens with a jump the robot must absorb.

    /opt/robot-platform/train-venv/bin/python anchor.py CKPT ROOT [--out X.json]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("dataset_root", type=Path)
p.add_argument("--train-root", type=Path, action="append", default=[])
p.add_argument("--stride", type=int, default=20)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--num-workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path)
args = p.parse_args()

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, "cuda")
is_patch = cfg.type == "patch_policy"
a_off = -min(list(cfg.action_delta_indices or [0]) + [0])
exclude = set().union(*(set(episode_action_hashes(r).values()) for r in args.train_root)) if args.train_root else None
repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
ds, meta, _ = build_dataset(args.dataset_root, repo_id, cfg, exclude, False)
idx = list(range(0, len(ds), args.stride))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

pol = demo = n = 0.0
per_dim_pol = per_dim_demo = None
with torch.no_grad():
    for batch in loader:
        gt = batch[ACTION][:, a_off].clone().float()
        state = batch[OBS_STATE].clone().float()
        if state.ndim == 3:
            state = state[:, -1]
        for cam in meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
        proc = preprocessor(batch)
        if is_patch:
            proc[OBS_IMAGES] = torch.stack([proc[k] for k in cfg.image_features], dim=-4)
            torch.manual_seed(args.seed)
            pred = policy.model.predict(proc)
        else:
            pred = policy.predict_action_chunk(proc)
        first = postprocessor(pred).float().cpu()[:, 0]
        dp, dd = (first - state).abs(), (gt - state).abs()
        pol += dp.mean().item() * len(first)
        demo += dd.mean().item() * len(first)
        per_dim_pol = dp.sum(0) if per_dim_pol is None else per_dim_pol + dp.sum(0)
        per_dim_demo = dd.sum(0) if per_dim_demo is None else per_dim_demo + dd.sum(0)
        n += len(first)

names = meta.info["features"]["action"]["names"]
res = {"checkpoint": str(args.checkpoint), "policy_type": cfg.type, "dataset": repo_id,
       "n_anchors": int(n),
       "policy_mae": round(pol / n, 6), "demonstration_mae": round(demo / n, 6),
       "ratio_policy_over_demo": round(pol / max(demo, 1e-9), 3),
       "per_dim": {k: {"policy": round(float(a) / n, 6), "demo": round(float(b) / n, 6),
                       "ratio": round(float(a) / max(float(b), 1e-9), 3)}
                   for k, a, b in zip(names, per_dim_pol, per_dim_demo)}}
print(json.dumps(res, indent=2))
if args.out:
    args.out.write_text(json.dumps(res, indent=2))
