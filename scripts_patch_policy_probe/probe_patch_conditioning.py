#!/usr/bin/env python
"""What does this `patch_policy` checkpoint actually condition on?

The checkpoint runs with `use_robot_state=False` -- the reference default -- so images are the
only input. That makes two questions decisive:

  1. **Which camera carries the signal?** Each camera is independently swapped for another
     anchor's frame and the emitted chunk is re-measured. The diffusion sampler is re-seeded
     before every call, so the initial noise and every per-step noise draw are identical across
     interventions and the whole delta is attributable to the intervention.
  2. **Can a purely visual policy locate the arm?** With no proprioception the chunk's first
     action has to be inferred from pixels. `first_action_vs_state` compares it to the measured
     pose the action should start from; the demonstrations' own first action is the ceiling.

Deltas are reported in raw joint units and as a fraction of `interframe_scale`, the mean absolute
difference between the chunks of two different anchors -- i.e. "how much do two unrelated
situations differ", so a delta near that scale means the pathway is fully used and a delta near
zero means it is dead.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_act_eval_test"))
from offline_chunk_eval import action_stats, build_dataset, episode_action_hashes, load_policy  # noqa: E402


class _PoolTokens(torch.nn.Module):
    """Wrap the frozen encoder so each frame's patch grid collapses to its own mean.

    This is Table 4's `n=1` rung applied at *inference* time: the token count, the mask and
    every learned weight stay exactly as trained, and only the spatial variation inside each
    frame is removed. If the emitted chunk barely moves, the dense grid this policy exists to
    provide is not being used on this task.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        out = self.inner(x)
        return out.mean(dim=-2, keepdim=True).expand_as(out)


@torch.no_grad()
def chunk(policy, cfg, batch, seed: int):
    from lerobot.utils.constants import OBS_IMAGES

    b = dict(batch)
    b[OBS_IMAGES] = torch.stack([b[key] for key in cfg.image_features], dim=-4)
    torch.manual_seed(seed)  # identical sampler noise across interventions
    return policy.model.predict(b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--keep-only-contaminated", action="store_true")
    p.add_argument("--n-anchors", type=int, default=48)
    p.add_argument("--stride", type=int, default=997)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from lerobot.utils.constants import ACTION, OBS_STATE

    policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, args.device)
    cams = list(cfg.image_features)
    start = cfg.n_obs_steps - 1

    exclude = set(episode_action_hashes(args.train_root).values()) if args.train_root else None
    repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
    ds, meta, _ = build_dataset(args.dataset_root, repo_id, cfg, exclude, args.keep_only_contaminated)

    idx = list(range(0, len(ds), args.stride))[: args.n_anchors]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    interventions = [
        "baseline",
        "all_cams_swapped",
        *[f"swap_{c.split('.')[-1]}" for c in cams],
        "history_frozen",       # frames 0..T-2 replaced by a copy of the newest frame
        "history_reversed",     # time order flipped: tests whether the history is read as a sequence
        "gray_images",          # every pixel 0.5 -- the unconditional prior of this model
        "patch_avg_pool",       # dense grid -> one token per frame, at inference only
        "quarter_resolution",   # 224 -> 56 -> 224: is fine spatial detail used at all?
    ]
    sums = {k: 0.0 for k in interventions if k != "baseline"}
    n_chunks = 0
    first_err_policy = first_err_gt = 0.0
    cross_anchor = 0.0
    n_cross = 0
    a_mean, a_std = action_stats(postprocessor)
    prev_base = None

    for bi, batch in enumerate(loader):
        for cam in meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
        gt = batch[ACTION][:, start:].clone().float()
        state = batch[OBS_STATE][:, -1].clone().float()
        proc = preprocessor(batch)

        base = postprocessor(chunk(policy, cfg, proc, args.seed)).float().cpu()
        # A within-batch roll is the donor for every swap: same distribution, different situation.
        donor = {c: proc[c].roll(1, dims=0) for c in cams}

        for name in interventions[1:]:
            v = dict(proc)
            if name == "all_cams_swapped":
                v.update(donor)
            elif name.startswith("swap_"):
                c = next(x for x in cams if x.split(".")[-1] == name[5:])
                v[c] = donor[c]
            elif name == "history_frozen":
                for c in cams:
                    v[c] = proc[c][:, -1:].expand_as(proc[c]).contiguous()
            elif name == "history_reversed":
                for c in cams:
                    v[c] = proc[c].flip(1)
            elif name == "gray_images":
                for c in cams:
                    v[c] = torch.full_like(proc[c], 0.5)
            elif name == "quarter_resolution":
                for c in cams:
                    x = proc[c]
                    flat = x.reshape(-1, *x.shape[-3:])
                    small = torch.nn.functional.interpolate(flat, size=(56, 56), mode="bilinear",
                                                            align_corners=False)
                    v[c] = torch.nn.functional.interpolate(
                        small, size=x.shape[-2:], mode="bilinear", align_corners=False
                    ).reshape(x.shape)

            if name == "patch_avg_pool":
                inner = policy.model.encoder
                policy.model.encoder = _PoolTokens(inner)
                out = postprocessor(chunk(policy, cfg, v, args.seed)).float().cpu()
                policy.model.encoder = inner
            else:
                out = postprocessor(chunk(policy, cfg, v, args.seed)).float().cpu()
            sums[name] += (out - base).abs().mean().item() * base.shape[0]

        first_err_policy += (base[:, 0] - state).abs().mean().item() * base.shape[0]
        first_err_gt += (gt[:, 0] - state).abs().mean().item() * base.shape[0]
        if prev_base is not None and prev_base.shape == base.shape:
            cross_anchor += (base - prev_base).abs().mean().item() * base.shape[0]
            n_cross += base.shape[0]
        prev_base = base
        n_chunks += base.shape[0]
        print(f"batch {bi}: {n_chunks} anchors done", flush=True)

    scale = cross_anchor / max(n_cross, 1)
    res = {
        "checkpoint": str(args.checkpoint),
        "dataset": repo_id,
        "held_out": args.train_root is not None and not args.keep_only_contaminated,
        "n_anchors": n_chunks,
        "use_robot_state": cfg.use_robot_state,
        "interframe_scale_rad": round(scale, 6),
        "interventions": {
            k: {"delta_rad": round(v / n_chunks, 6),
                "frac_of_interframe": round((v / n_chunks) / scale, 4) if scale else None}
            for k, v in sums.items()
        },
        "first_action_vs_state": {
            "policy_mae_rad": round(first_err_policy / n_chunks, 6),
            "demonstration_mae_rad": round(first_err_gt / n_chunks, 6),
            "ratio_policy_over_demo": round(first_err_policy / max(first_err_gt, 1e-9), 3),
        },
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
