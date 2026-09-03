#!/usr/bin/env python
"""Score the trajectory the robot is actually commanded to follow, not the chunk the model emits.

`deploy_config_patch_policy.yaml` sets `n_action_steps: 8`, and
`lerobot_vlahost/src/lerobot/rollout/strategies/core.py::send_next_action_chunk` then sets
`bridge_steps = min(40, 8) = 8`. For the **14 arm joints** every commanded frame is therefore a
cubic Hermite that starts at the measured pose with zero velocity and ends at `chunk[7]`; the
model contributes one waypoint per joint per inference. The **two gripper channels are excluded
from the bridge** (`joint_count=14`) and pass through raw.

So neither `policy` nor `policy_reanchored` in `eval_patch_chunk.py` is what the robot executes.
This script builds the real thing, using the deploy repository's own `cubic_hermite_segment` so
the curve is bit-identical, and scores four conditions over the 8 executed frames:

  policy_bridge   the deployed trajectory
  oracle_bridge   the same bridge aimed at the *demonstrated* waypoint `gt[7]` -- the ceiling the
                  bridge design allows, i.e. how much is lost to the bridge itself rather than to
                  the model
  raw_chunk       the model's chunk with no bridge, for reference
  hold_state      do nothing

Arm joints and gripper channels are reported separately, because only the arms are bridged and
only the grippers decide whether the hand closes at the right moment.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_act_eval_test"))
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402

VLAHOST_TRAJECTORY = Path("/home/kewei/YING/lerobot_vlahost/src/lerobot/rollout/trajectory.py")


def load_hermite():
    """Import the deploy repo's trajectory module by path.

    Not `sys.path`-prepending the vlahost tree: this process already has a different `lerobot`
    installed (the one that trained the checkpoint), and shadowing it would silently swap the
    policy implementation under the evaluation.
    """
    spec = importlib.util.spec_from_file_location("_vlahost_trajectory", VLAHOST_TRAJECTORY)
    mod = importlib.util.module_from_spec(spec)
    # The module defines a @dataclass, and dataclasses resolves annotations through
    # sys.modules[cls.__module__] -- so it has to be registered before exec_module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.cubic_hermite_segment


def build_bridge(hermite, state, waypoint, nxt, k, n_arm=14):
    """(B, J) start / (B, J) target -> (B, k, J), arms bridged, grippers left to the caller."""
    b, j = state.shape
    out = torch.empty(b, k, j)
    for bi in range(b):
        for ji in range(n_arm):
            out[bi, :, ji] = hermite(
                float(state[bi, ji]), float(waypoint[bi, ji]), k,
                start_velocity=0.0,
                end_velocity=float(nxt[bi, ji] - waypoint[bi, ji]),
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, action="append", required=True)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--n-action-steps", type=int, default=8, help="deploy value, not the config's")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-anchors-per-dataset", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    hermite = load_hermite()
    policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, args.device)
    k = args.n_action_steps
    start = cfg.n_obs_steps - 1
    exclude = set(episode_action_hashes(args.train_root).values()) if args.train_root else None

    conds = ("policy_bridge", "oracle_bridge", "raw_chunk", "hold_state")
    arm = {c: 0.0 for c in conds}
    grip = {c: 0.0 for c in conds}
    n = 0
    t0 = time.time()

    for root in args.dataset_root:
        repo_id = f"{root.parent.name}/{root.name}"
        ds, meta, n_dropped = build_dataset(root, repo_id, cfg, exclude, False)
        if ds is None:
            print(f"[{repo_id}] SKIPPED (all {n_dropped} episodes are training data)", flush=True)
            continue
        idx = list(range(0, len(ds), args.stride))
        if args.max_anchors_per_dataset:
            idx = idx[: args.max_anchors_per_dataset]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
            shuffle=False, num_workers=4)

        for batch in loader:
            gt = batch[ACTION][:, start : start + k + 1].clone().float()
            state = batch[OBS_STATE][:, -1].clone().float()
            for cam in meta.camera_keys:
                if batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].float() / 255.0
            proc = preprocessor(batch)
            proc[OBS_IMAGES] = torch.stack([proc[key] for key in cfg.image_features], dim=-4)
            with torch.no_grad():
                pred = postprocessor(policy.model.predict(proc)).float().cpu()

            traj = {}
            # The bridge overwrites arms only; grippers keep whatever that condition supplies.
            traj["policy_bridge"] = build_bridge(hermite, state, pred[:, k - 1], pred[:, k], k)
            traj["policy_bridge"][:, :, 14:] = pred[:, :k, 14:]
            traj["oracle_bridge"] = build_bridge(hermite, state, gt[:, k - 1], gt[:, k], k)
            traj["oracle_bridge"][:, :, 14:] = gt[:, :k, 14:]
            traj["raw_chunk"] = pred[:, :k]
            traj["hold_state"] = state.unsqueeze(1).expand(-1, k, -1)

            tgt = gt[:, :k]
            b = tgt.shape[0]
            for c in conds:
                arm[c] += (traj[c][:, :, :14] - tgt[:, :, :14]).abs().mean().item() * b
                grip[c] += (traj[c][:, :, 14:] - tgt[:, :, 14:]).abs().mean().item() * b
            n += b
        print(f"[{repo_id}] {n} anchors ({time.time() - t0:.0f}s)", flush=True)

    res = {
        "checkpoint": str(args.checkpoint),
        "n_action_steps_deployed": k,
        "anchors": n,
        "note": "arm joints are Hermite-bridged from the measured pose; grippers pass through raw",
        "arm_mae_rad": {c: round(arm[c] / n, 6) for c in conds},
        "gripper_mae": {c: round(grip[c] / n, 6) for c in conds},
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
