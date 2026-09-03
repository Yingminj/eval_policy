#!/usr/bin/env python
"""Offline action-chunk evaluation of a trained LeRobot checkpoint on held-out datasets.

Why this exists
---------------
The training runs on this cluster log only a training loss, and each policy family
optimises a *different* objective (ACT's L1+KL, act_dit's velocity-MSE, patch_policy's
epsilon-MSE).  Those numbers are not comparable to each other and say nothing about
generalisation.  This script produces one quantity that is defined identically for every
policy: the error between the action chunk the policy actually *emits* and the
demonstrated action, measured on episodes the checkpoint never trained on, in physical
joint units.

Method
------
For every sampled anchor frame t of a held-out episode:
  1. Build the observation exactly as training did (same `delta_timestamps`, same
     uint8->float/255 conversion, same preprocessor loaded *from the checkpoint*, so the
     normalisation statistics are the training-set statistics -- what deployment uses).
  2. Call `policy.predict_action_chunk`, which returns the full chunk the policy would
     emit from that single observation (open loop, no ground truth fed back).
  3. Un-normalise with the checkpoint's own postprocessor -> raw joint units.
  4. Compare against the recorded action at t..t+chunk_size-1, masking frames that
     `action_is_pad` marks as past the end of the episode.

Two reference predictors are scored on exactly the same anchors and mask, because an
absolute error in radians is uninterpretable on its own:
  * `hold_state`  - emit the current measured joint state for the whole chunk ("do
                    nothing, hold the current pose").  State and action share the same
                    16-joint layout in this profile, so this is the honest null policy.
  * `train_mean`  - emit the training-set mean action for the whole chunk.
A policy that cannot beat `hold_state` has not learned the task, however low its
training loss went.

Normalised errors divide by the training-set per-joint action std (the same statistic
the training loss used), so `norm_mae` at the full horizon is directly comparable to the
`l1_loss` printed in the training log -- ACT in eval mode uses a zero VAE latent, so
`predict_action_chunk` and `forward()` share one forward pass.

Usage
-----
    python offline_chunk_eval.py \
        --checkpoint /mnt/robot_platform/jobs/<job>/run/checkpoints/last/pretrained_model \
        --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_1 \
        --dataset-root ... \
        --stride 20 --batch-size 16 --out report.json

Self-check:  python offline_chunk_eval.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------------------
# metric accumulation
# --------------------------------------------------------------------------------------
class ChunkErrorAccumulator:
    """Streaming sum of |pred - gt| and (pred - gt)^2 over (horizon, joint), masked.

    Kept as running sums rather than a list of batches so memory does not grow with the
    number of anchors -- a full pass over six datasets is ~10^4 anchors x 100 x 16.
    """

    def __init__(self, horizon: int, n_joints: int, device: str = "cpu"):
        self.abs_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.sq_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.count = torch.zeros(horizon, 1, dtype=torch.float64, device=device)

    def update(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> None:
        """pred/gt: (B, H, J) float. valid: (B, H) bool -- False where the chunk ran past
        the end of the episode and the recorded action is padding."""
        err = (pred.double() - gt.double()) * valid.unsqueeze(-1).double()
        self.abs_sum += err.abs().sum(dim=0)
        self.sq_sum += err.pow(2).sum(dim=0)
        self.count += valid.double().sum(dim=0).unsqueeze(-1)

    # -- reductions ---------------------------------------------------------------------
    def _safe(self, num: torch.Tensor) -> torch.Tensor:
        return num / self.count.clamp_min(1.0)

    def mae_per_horizon_joint(self) -> torch.Tensor:
        return self._safe(self.abs_sum)

    def mae_per_horizon(self) -> torch.Tensor:
        return self.mae_per_horizon_joint().mean(dim=1)

    def mae_per_joint(self, upto: int | None = None) -> torch.Tensor:
        h = slice(0, upto) if upto else slice(None)
        return self.abs_sum[h].sum(0) / self.count[h].sum().clamp_min(1.0)

    def mae(self, upto: int | None = None) -> float:
        h = slice(0, upto) if upto else slice(None)
        n = self.count[h].sum() * self.abs_sum.shape[1]
        return (self.abs_sum[h].sum() / n.clamp_min(1.0)).item()

    def rmse(self, upto: int | None = None) -> float:
        h = slice(0, upto) if upto else slice(None)
        n = self.count[h].sum() * self.sq_sum.shape[1]
        return (self.sq_sum[h].sum() / n.clamp_min(1.0)).sqrt().item()

    def n_anchor_steps(self) -> int:
        return int(self.count.sum().item())


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device

    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    # The processors are loaded *from the checkpoint*, so the normalisation statistics are
    # the ones the model was trained with. Recomputing them on the eval set would leak
    # eval-set statistics into the model's inputs and flatter the result.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def episode_action_hashes(root: Path) -> dict[int, str]:
    """SHA1 of each episode's raw action array, as an identity fingerprint.

    Needed because the dataset names in this corpus lie about provenance: the batches are
    *cumulative* merges, so `batch_5` and `batch_6` are wholly contained in the training
    set `batch_success_361` and `batch_4` is 38% contained, despite being separate
    directories with different episode counts. Matching on episode counts or on episode
    lengths finds false negatives; only the data itself is conclusive.
    """
    import glob

    import numpy as np
    import pandas as pd

    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    out = {}
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(np.float32)
        out[int(ep)] = hashlib.sha1(a.tobytes()).hexdigest()
    return out


def build_dataset(root: Path, repo_id: str, cfg, exclude_hashes: set[str] | None, invert: bool = False):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    # Same helper the training script uses, so the batch layout is identical to training.
    delta_timestamps = resolve_delta_timestamps(cfg, meta)

    episodes = None
    n_dropped = 0
    if exclude_hashes:
        fp = episode_action_hashes(root)
        # invert=True keeps ONLY the contaminated episodes. That is the within-session
        # control: scoring the seen and unseen halves of one recording batch holds the
        # session, lighting and object layout fixed, so the difference between them is
        # memorisation alone rather than memorisation plus distribution shift.
        episodes = sorted(
            ep for ep, h in fp.items() if (h in exclude_hashes) == invert
        )
        n_dropped = len(fp) - len(episodes)
        if not episodes:
            return None, meta, n_dropped

    ds = LeRobotDataset(
        repo_id=repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps
    )
    return ds, meta, n_dropped


def action_stats(postprocessor) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-set action mean/std, read out of the checkpoint's unnormaliser."""
    from lerobot.utils.constants import ACTION

    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats and ACTION in stats:
            s = stats[ACTION]
            as_t = lambda x: torch.as_tensor(x, dtype=torch.float32).flatten().cpu()  # noqa: E731
            return as_t(s["mean"]), as_t(s["std"])
    raise RuntimeError("no action mean/std found in the postprocessor pipeline")


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    dataset_roots: list[Path],
    stride: int,
    batch_size: int,
    num_workers: int,
    device: str,
    max_anchors_per_dataset: int | None,
    train_root: Path | None,
    invert_filter: bool = False,
    dump_traces: Path | None = None,
    trace_anchors: int = 40,
    trace_episodes: set[int] | None = None,
) -> dict:
    from lerobot.utils.constants import ACTION, OBS_STATE

    policy, preprocessor, postprocessor, cfg = load_policy(checkpoint, device)
    horizon = cfg.chunk_size

    exclude_hashes = None
    if train_root is not None:
        exclude_hashes = set(episode_action_hashes(train_root).values())
        print(f"contamination filter: {len(exclude_hashes)} training episodes fingerprinted "
              f"from {train_root.name}", flush=True)
    a_mean, a_std = action_stats(postprocessor)

    per_dataset: dict[str, dict] = {}
    overall = {k: None for k in ("policy", "hold_state", "train_mean")}
    n_joints = None
    joint_names = None
    t_start = time.time()
    # Raw per-anchor predictions, kept only for the first scored dataset and only for the
    # first `trace_anchors` anchors (or only for the requested `trace_episodes`): the
    # accumulator above is streaming by design, so the arrays a trajectory plot needs do
    # not otherwise survive the loop.
    traces: dict[str, list] | None = {} if dump_traces else None
    traces_n = 0

    for root in dataset_roots:
        repo_id = f"{root.parent.name}/{root.name}"
        ds, meta, n_dropped = build_dataset(root, repo_id, cfg, exclude_hashes, invert_filter)
        if ds is None:
            print(f"[{repo_id}] SKIPPED - all {n_dropped} episodes are in the training set", flush=True)
            per_dataset[repo_id] = {"skipped": "fully contained in training set",
                                    "episodes_dropped": n_dropped}
            continue
        if n_joints is None:
            n_joints = meta.features[ACTION]["shape"][0]
            joint_names = meta.features[ACTION].get("names") or [f"j{i}" for i in range(n_joints)]
            for k in overall:
                overall[k] = ChunkErrorAccumulator(horizon, n_joints)

        idx = list(range(0, len(ds), stride))
        if max_anchors_per_dataset:
            idx = idx[:max_anchors_per_dataset]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )

        acc = {k: ChunkErrorAccumulator(horizon, n_joints) for k in overall}
        camera_keys = meta.camera_keys
        t0 = time.time()

        for batch in loader:
            # Ground truth and state are snapshotted BEFORE the preprocessor runs: the
            # normaliser rewrites batch[ACTION] in place, and we want raw joint units.
            gt = batch[ACTION].clone().float()                 # (B, H, J)
            state = batch[OBS_STATE].clone().float()           # (B, J)
            valid = ~batch["action_is_pad"]                    # (B, H)

            # Mirror lerobot_train.py's eval path: uint8 frames are scaled to [0,1]
            # before the normaliser sees them.
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0

            processed = preprocessor(batch)
            pred = policy.predict_action_chunk(processed)      # normalised, (B, H, J)
            pred = postprocessor(pred).float().cpu()           # raw joint units

            acc["policy"].update(pred, gt, valid)
            acc["hold_state"].update(state.unsqueeze(1).expand_as(gt), gt, valid)
            acc["train_mean"].update(
                a_mean.view(1, 1, -1).expand_as(gt), gt, valid
            )

            if traces is not None and traces_n < trace_anchors:
                if trace_episodes is not None:
                    keep = torch.isin(
                        batch["episode_index"].cpu(), torch.tensor(sorted(trace_episodes))
                    )
                    n_keep = int(keep.sum())
                else:
                    keep, n_keep = None, len(gt)
                if n_keep:
                    for key, val in (
                        ("pred", pred), ("gt", gt), ("state", state), ("valid", valid),
                        ("episode_index", batch["episode_index"]), ("frame_index", batch["frame_index"]),
                    ):
                        arr = val.cpu().numpy()
                        traces.setdefault(key, []).append(
                            arr[keep.numpy()] if keep is not None else arr
                        )
                    traces_n += n_keep

        if traces:
            import numpy as np
            dump_traces.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dump_traces,
                joint_names=np.array(joint_names),
                repo_id=repo_id,
                **{k: np.concatenate(v)[:trace_anchors] for k, v in traces.items()},
            )
            print(f"wrote {dump_traces}  ({traces_n} anchors from {repo_id})", flush=True)
            traces = None

        for k in overall:
            overall[k].abs_sum += acc[k].abs_sum
            overall[k].sq_sum += acc[k].sq_sum
            overall[k].count += acc[k].count

        per_dataset[repo_id] = {
            "episodes_evaluated": meta.total_episodes - n_dropped,
            "episodes_dropped_as_contaminated": n_dropped,
            "frames": meta.total_frames,
            "anchors": len(idx),
            "anchor_action_steps": acc["policy"].n_anchor_steps(),
            "seconds": round(time.time() - t0, 1),
            **{k: summarise(acc[k], a_std, horizon, joint_names) for k in overall},
        }
        print(
            f"[{repo_id}] {len(idx)} anchors  "
            f"policy_mae={acc['policy'].mae():.5f}  "
            f"hold_state_mae={acc['hold_state'].mae():.5f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    return {
        "checkpoint": str(checkpoint),
        "policy_type": cfg.type,
        "chunk_size": horizon,
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "joint_names": joint_names,
        "stride": stride,
        "device": device,
        "total_seconds": round(time.time() - t_start, 1),
        "per_dataset": per_dataset,
        "aggregate": {k: summarise(overall[k], a_std, horizon, joint_names) for k in overall},
    }


def summarise(acc: ChunkErrorAccumulator, a_std: torch.Tensor, horizon: int, names: list[str]) -> dict:
    """Raw-unit and std-normalised reductions of one accumulator."""
    mae_hj = acc.mae_per_horizon_joint()
    norm_hj = mae_hj / a_std.double().clamp_min(1e-8)
    cuts = [c for c in (1, 10, 25, 50, horizon) if c <= horizon]
    return {
        "mae": acc.mae(),
        "rmse": acc.rmse(),
        "norm_mae": norm_hj.mean().item(),
        "mae_at_horizon": {str(c): acc.mae(upto=c) for c in cuts},
        "norm_mae_at_horizon": {str(c): norm_hj[:c].mean().item() for c in cuts},
        "mae_per_horizon": [round(v, 6) for v in acc.mae_per_horizon().tolist()],
        "mae_per_joint": dict(zip(names, [round(v, 6) for v in acc.mae_per_joint().tolist()])),
        "norm_mae_per_joint": dict(zip(names, [round(v, 6) for v in norm_hj.mean(0).tolist()])),
        "anchor_action_steps": acc.n_anchor_steps(),
    }


# --------------------------------------------------------------------------------------
def selftest() -> None:
    """The accumulator is the only non-trivial logic here; check it against a hand case."""
    acc = ChunkErrorAccumulator(horizon=3, n_joints=2)
    pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]])   # (1,3,2)
    gt = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])                    # last step is padding
    acc.update(pred, gt, valid)

    # padded step must contribute nothing, to neither the sum nor the count
    assert acc.count.flatten().tolist() == [1.0, 1.0, 0.0], acc.count
    assert abs(acc.mae_per_horizon()[2].item()) < 1e-12, "padding leaked into the mean"
    # mean over the two valid steps and two joints: (1+1+2+2)/4 = 1.5
    assert abs(acc.mae() - 1.5) < 1e-12, acc.mae()
    # horizon-1 cut sees only the first step: (1+1)/2 = 1.0
    assert abs(acc.mae(upto=1) - 1.0) < 1e-12, acc.mae(upto=1)
    # rmse over valid: sqrt((1+1+4+4)/4) = sqrt(2.5)
    assert abs(acc.rmse() - 2.5**0.5) < 1e-12, acc.rmse()

    # two batches must accumulate exactly like one concatenated batch
    a, b = ChunkErrorAccumulator(2, 1), ChunkErrorAccumulator(2, 1)
    p1, g1 = torch.tensor([[[1.0], [3.0]]]), torch.zeros(1, 2, 1)
    p2, g2 = torch.tensor([[[5.0], [7.0]]]), torch.zeros(1, 2, 1)
    v = torch.ones(1, 2, dtype=torch.bool)
    a.update(p1, g1, v)
    a.update(p2, g2, v)
    b.update(torch.cat([p1, p2]), torch.cat([g1, g2]), torch.cat([v, v]))
    assert torch.allclose(a.mae_per_horizon(), b.mae_per_horizon())
    assert abs(a.mae() - 4.0) < 1e-12, a.mae()
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset-root", type=Path, action="append", default=[])
    p.add_argument("--stride", type=int, default=20, help="sample every Nth frame as a chunk anchor")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-anchors-per-dataset", type=int, default=None)
    p.add_argument("--train-root", type=Path, default=None,
                   help="training dataset; its episodes are fingerprinted and any\neval episode with matching action data is dropped")
    p.add_argument("--keep-only-contaminated", action="store_true",
                   help="invert --train-root: score ONLY episodes that ARE in the\ntraining set (within-session control)")
    p.add_argument("--out", type=Path)
    p.add_argument("--dump-traces", type=Path, default=None,
                   help="save raw per-anchor pred/gt/state for the first scored dataset to an .npz, "
                        "for plot_traces.py")
    p.add_argument("--trace-anchors", type=int, default=200,
                   help="max anchors to keep in the trace dump (default 40)")
    p.add_argument("--trace-episode", type=int, action="append", default=[],
                   help="only dump anchors from this episode index (repeatable). Without it the "
                        "first --trace-anchors anchors of the dataset are kept, which with "
                        "stride>1 all come from the first episode -- pass --trace-episode N "
                        "(and raise --trace-anchors if N's episode has more anchors) to plot "
                        "a specific episode")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    # --checkpoint is the pretrained_model/ dir, not the job dir. Getting this wrong throws
    # a draccus ParsingError from deep inside config decoding, which says nothing useful.
    if args.checkpoint and not (args.checkpoint / "config.json").is_file():
        found = sorted(args.checkpoint.glob("run/checkpoints/*/pretrained_model"))
        hint = f"\n  did you mean:  --checkpoint {found[-1]}" if found else ""
        p.error(f"no config.json in {args.checkpoint} -- point --checkpoint at a "
                f"pretrained_model/ directory.{hint}")
    if args.dump_traces and args.dump_traces.is_dir():
        p.error(f"--dump-traces must be a file path, not a directory; "
                f"try {args.dump_traces / 'traces.npz'}")

    if not args.checkpoint or not args.dataset_root:
        p.error("--checkpoint and at least one --dataset-root are required")

    selftest()  # cheap; never report numbers from a build whose accumulator is broken
    report = evaluate(
        checkpoint=args.checkpoint,
        dataset_roots=args.dataset_root,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_anchors_per_dataset=args.max_anchors_per_dataset,
        train_root=args.train_root,
        invert_filter=args.keep_only_contaminated,
        dump_traces=args.dump_traces,
        trace_anchors=args.trace_anchors,
        trace_episodes=set(args.trace_episode) or None,
    )
    agg = report["aggregate"]
    print("\n=== aggregate over all held-out datasets ===")
    for k, v in agg.items():
        print(f"  {k:<12} mae={v['mae']:.5f}  rmse={v['rmse']:.5f}  norm_mae={v['norm_mae']:.5f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
