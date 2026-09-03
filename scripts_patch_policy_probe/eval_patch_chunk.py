#!/usr/bin/env python
"""Offline chunk evaluation for `patch_policy`, against the same null baselines as
`../scripts_act_eval_test/offline_chunk_eval.py`.

`patch_policy` differs from `act`/`act_dit` in three ways that the original harness cannot
absorb, which is why this wrapper exists rather than a flag:

  * the chunk length is `action_chunk_size`, not `chunk_size`;
  * `n_obs_steps > 1`, so the dataloader hands over `(B, T, ...)` observations and
    `n_obs_steps + action_chunk_size - 1` actions -- the predicted chunk lines up with
    `action[:, n_obs_steps - 1:]`, not with `action[:, 0:]`;
  * `predict_action_chunk` reads from the online queues, so the offline path calls
    `policy.model.predict` directly (the same thing minus the `n_action_steps` slice, so
    the whole horizon can be scored).

Everything else -- contamination fingerprinting, the `hold_state` / `train_mean` baselines,
the error accumulator -- is imported from the ACT harness so the two reports' numbers are
produced by the same code.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_act_eval_test"))
from offline_chunk_eval import (  # noqa: E402
    ChunkErrorAccumulator,
    action_stats,
    build_dataset,
    episode_action_hashes,
    load_policy,
    summarise,
)


@torch.no_grad()
def evaluate(args) -> dict:
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, args.device)
    if cfg.type != "patch_policy":
        raise SystemExit(f"this harness is for patch_policy, got {cfg.type}")

    horizon = cfg.action_chunk_size
    start = cfg.n_obs_steps - 1          # predicted chunk starts at the newest observation
    if args.num_inference_steps:
        policy.model.num_inference_steps = args.num_inference_steps

    exclude = None
    if args.train_root:
        exclude = set(episode_action_hashes(args.train_root).values())
        print(f"contamination filter: {len(exclude)} training episodes fingerprinted", flush=True)
    a_mean, a_std = action_stats(postprocessor)

    per_dataset, n_joints, joint_names = {}, None, None
    # `policy_reanchored` is not a null baseline: it is the deployed policy. With
    # `n_action_steps: 8` the deploy path sets `bridge_steps = min(40, 8) = 8`, so every frame
    # actually sent to the robot is a cubic Hermite from the *measured* pose to `chunk[-1]`.
    # Re-anchoring the predicted chunk onto the measured pose reproduces that, and the gap
    # between `policy` and `policy_reanchored` separates "does not know where the arm is"
    # from "does not know what motion to make".
    overall = {k: None for k in ("policy", "policy_reanchored", "hold_state", "train_mean")}
    t_start = time.time()

    for root in args.dataset_root:
        repo_id = f"{root.parent.name}/{root.name}"
        ds, meta, n_dropped = build_dataset(root, repo_id, cfg, exclude, args.keep_only_contaminated)
        if ds is None:
            print(f"[{repo_id}] SKIPPED - all {n_dropped} episodes are in the training set", flush=True)
            per_dataset[repo_id] = {"skipped": "fully contained in training set"}
            continue
        if n_joints is None:
            n_joints = meta.features[ACTION]["shape"][0]
            joint_names = meta.features[ACTION].get("names") or [f"j{i}" for i in range(n_joints)]
            for k in overall:
                overall[k] = ChunkErrorAccumulator(horizon, n_joints)

        idx = list(range(0, len(ds), args.stride))
        if args.max_anchors_per_dataset:
            idx = idx[: args.max_anchors_per_dataset]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )

        acc = {k: ChunkErrorAccumulator(horizon, n_joints) for k in overall}
        camera_keys = meta.camera_keys
        t0 = time.time()

        for batch in loader:
            gt = batch[ACTION][:, start:].clone().float()            # (B, chunk, J)
            state = batch[OBS_STATE][:, -1].clone().float()          # newest frame's pose
            valid = ~batch["action_is_pad"][:, start:]

            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0

            processed = preprocessor(batch)
            processed[OBS_IMAGES] = torch.stack(
                [processed[key] for key in cfg.image_features], dim=-4
            )
            pred = policy.model.predict(processed)                   # (B, chunk, J), normalised
            pred = postprocessor(pred).float().cpu()

            reanchored = pred - pred[:, :1] + state.unsqueeze(1)
            acc["policy"].update(pred, gt, valid)
            acc["policy_reanchored"].update(reanchored, gt, valid)
            acc["hold_state"].update(state.unsqueeze(1).expand_as(gt), gt, valid)
            acc["train_mean"].update(a_mean.view(1, 1, -1).expand_as(gt), gt, valid)

        for k in overall:
            overall[k].abs_sum += acc[k].abs_sum
            overall[k].sq_sum += acc[k].sq_sum
            overall[k].count += acc[k].count

        per_dataset[repo_id] = {
            "episodes_evaluated": meta.total_episodes - n_dropped,
            "episodes_dropped_as_contaminated": n_dropped,
            "anchors": len(idx),
            "seconds": round(time.time() - t0, 1),
            **{k: summarise(acc[k], a_std, horizon, joint_names) for k in overall},
        }
        print(
            f"[{repo_id}] {len(idx)} anchors  policy_mae={acc['policy'].mae():.5f}  "
            f"hold_state_mae={acc['hold_state'].mae():.5f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    return {
        "checkpoint": str(args.checkpoint),
        "policy_type": cfg.type,
        "action_head": cfg.action_head,
        "vision_encoder": cfg.vision_encoder,
        "use_robot_state": cfg.use_robot_state,
        "n_obs_steps": cfg.n_obs_steps,
        "action_chunk_size": horizon,
        "n_action_steps": cfg.n_action_steps,
        "num_inference_steps": getattr(policy.model, "num_inference_steps", None),  # act head has none
        "joint_names": joint_names,
        "stride": args.stride,
        "total_seconds": round(time.time() - t_start, 1),
        "per_dataset": per_dataset,
        "aggregate": {k: summarise(overall[k], a_std, horizon, joint_names) for k in overall},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, action="append", required=True)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-anchors-per-dataset", type=int, default=None)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--keep-only-contaminated", action="store_true")
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    res = evaluate(args)
    args.out.write_text(json.dumps(res, indent=2))
    agg = res["aggregate"]
    print(f"\npolicy      MAE {agg['policy']['mae']:.5f}")
    print(f"hold_state  MAE {agg['hold_state']['mae']:.5f}")
    print(f"train_mean  MAE {agg['train_mean']['mae']:.5f}")
    print(f"-> written to {args.out}")


if __name__ == "__main__":
    main()
