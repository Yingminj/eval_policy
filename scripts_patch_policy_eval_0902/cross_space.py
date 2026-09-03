#!/usr/bin/env python
"""Joint-space and EEF-space policies scored on the same physical quantity.

The 08-31 report could only compare the two action spaces through each one's own null
ratio, because "MAE in radians" and "MAE in metres" are not the same number.  That is no
longer necessary: `fk_provenance.txt` shows the EEF datasets are bit-exact forward
kinematics of the joint datasets (`tool/tr_joint_to_eef.py`, M6-696 MJCF, left_tool /
right_tool frames).  So a joint-space chunk can be pushed through the same FK and scored
in metres and radians against the same ground truth as an EEF-space chunk.

Every run scores the same 53 recordings at the same stride, so the anchors line up; a
joint policy is fed the 16-D dataset and an EEF policy the 14-D one, and both come out in
the 14-D EEF frame.  `hold_state` (the arm stays where it is) is identical for both by
construction, which is what makes the two blocks commensurable.

    /opt/robot-platform/train-venv/bin/python cross_space.py NAME CKPT ROOT [--train-root R] --out X.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/kewei/YING/robot_data_platform/tool")
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402
from tr_joint_to_eef import MjcfForwardKinematics  # noqa: E402

MJCF = "/home/kewei/YING/Apex_Deploy_new/robot_node/marvin_description/mjcf/matrix/m6_696.xml"
GRIPPERS = ["gripper_L", "gripper_R"]
GROUPS = {"position_m": list(range(0, 3)) + list(range(6, 9)),
          "rotation_rad": list(range(3, 6)) + list(range(9, 12)),
          "gripper": [12, 13]}

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("checkpoint", type=Path)
p.add_argument("dataset_root", type=Path)
p.add_argument("--train-root", type=Path, action="append", default=[])
p.add_argument("--stride", type=int, default=20)
p.add_argument("--horizon", type=int, default=50)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--num-workers", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, "cuda")
is_patch = cfg.type == "patch_policy"
a_off = -min(list(cfg.action_delta_indices or [0]) + [0])
exclude = set().union(*(set(episode_action_hashes(r).values()) for r in args.train_root)) if args.train_root else None
repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
ds, meta, _ = build_dataset(args.dataset_root, repo_id, cfg, exclude, False)
names = meta.info["features"]["action"]["names"]
joint_space = len(names) != 14 or not names[0].startswith("eef_")
fk = MjcfForwardKinematics(MJCF, names, "left_tool", "right_tool", "euler") if joint_space else None
gi = [names.index(g) for g in GRIPPERS] if joint_space else None


def to_eef(x: torch.Tensor) -> np.ndarray:
    """(B, T, J) in the dataset's own action space -> (B, T, 14) EEF, exactly as tr_joint_to_eef does."""
    a = x.numpy()
    if not joint_space:
        return a
    flat = a.reshape(-1, a.shape[-1]).astype(np.float64)
    left, right = fk.evaluate(flat)
    out = np.concatenate([left, right, flat[:, gi].astype(np.float32)], axis=1)
    return out.reshape(*a.shape[:2], 14)


idx = list(range(0, len(ds), args.stride))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)
acc = {k: np.zeros(14) for k in ("policy", "hold_state")}
cuts = (1, 10, 25, 50)
cut_err = {k: {h: 0.0 for h in cuts} for k in ("policy", "hold_state")}
cut_n = {h: 0 for h in cuts}
n = n_anchors = 0
with torch.no_grad():
    for batch in loader:
        gt = to_eef(batch[ACTION][:, a_off:a_off + args.horizon].clone().float())
        valid = (~batch["action_is_pad"][:, a_off:a_off + args.horizon]).numpy()
        state = batch[OBS_STATE].clone().float()
        if state.ndim == 3:
            state = state[:, -1]
        hold = to_eef(state[:, None].expand(-1, args.horizon, -1).contiguous())
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
        pred = to_eef(postprocessor(pred).float().cpu()[:, :args.horizon])
        m = valid[..., None]
        for key, arr in (("policy", pred), ("hold_state", hold)):
            d = np.abs(arr - gt) * m
            acc[key] += d.sum((0, 1))
            for h in cuts:
                cut_err[key][h] += d[:, :h].sum()
        for h in cuts:
            cut_n[h] += int(m[:, :h].sum()) * 14
        n += int(m.sum())
        n_anchors += len(gt)

res = {"name": args.name, "checkpoint": str(args.checkpoint), "policy_type": cfg.type,
       "dataset": repo_id, "source_space": "joint" if joint_space else "eef",
       "horizon": args.horizon, "n_anchors": n_anchors, "valid_steps": n}
for key in ("policy", "hold_state"):
    per_dim = acc[key] / n
    res[key] = {
        "mae": float(per_dim.mean()),
        **{g: float(per_dim[ix].mean()) for g, ix in GROUPS.items()},
        "position_mm": float(per_dim[GROUPS["position_m"]].mean() * 1000),
        "rotation_deg": float(np.rad2deg(per_dim[GROUPS["rotation_rad"]].mean())),
        "mae_at_horizon": {str(h): float(cut_err[key][h]) / cut_n[h] for h in cuts},
        "per_dim": {k: float(v) for k, v in zip(
            [f"eef_{s}_{a}" for s in "lr" for a in ("x", "y", "z", "roll", "pitch", "yaw")] + GRIPPERS,
            per_dim)},
    }
res["vs_null_position"] = res["hold_state"]["position_m"] / res["policy"]["position_m"]
res["vs_null_rotation"] = res["hold_state"]["rotation_rad"] / res["policy"]["rotation_rad"]
print(json.dumps({k: v for k, v in res.items() if k != "per_dim"}, indent=2, default=str)[:2000])
args.out.write_text(json.dumps(res, indent=2, default=float))
