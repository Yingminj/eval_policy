#!/usr/bin/env python
"""Can the arm's pose be read out of the frozen encoder's patch tokens at all?

This checkpoint runs `use_robot_state=False` (the reference default), so the only way it can
place an action chunk at the arm's current pose is to infer that pose from pixels. If the
frozen DINOv2 tokens do not carry the pose, no head stacked on them can recover it, and every
re-plan starts from the wrong place -- which is a grasp failure regardless of how the action
head is tuned.

Two readouts are fitted on training frames and scored on held-out frames:

  * `mean_pool`  -- ridge regression on each camera's mean-pooled token (3 x 384 features).
    This is the "global representation" the paper argues against.
  * `all_patches` -- a single linear layer over every patch token (3 x 256 x 384 features),
    trained with Adam. This is the most a *linear* readout of the dense grid can do, so it
    upper-bounds what the policy's first decoder layer can extract from one frame.

Both are compared against `train_mean` (predict the training-set mean pose) and reported in raw
joint units, so the numbers are directly comparable to the chunk MAE in the other scripts here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_act_eval_test"))
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402


@torch.no_grad()
def extract(policy, preprocessor, cfg, ds, idx, batch_size, device, tag):
    from lerobot.utils.constants import OBS_STATE

    feats, states = [], []
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=4
    )
    t0 = time.time()
    for i, batch in enumerate(loader):
        # Only the newest observation step is needed: this asks what one frame reveals.
        state = batch[OBS_STATE]
        states.append((state[:, -1] if state.ndim == 3 else state).clone().float())
        per_cam = []
        for key in cfg.image_features:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1]
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            per_cam.append(policy.model.encoder(img.to(device)))  # (B, P, E)
        feats.append(torch.cat(per_cam, dim=1).half().cpu())      # (B, V*P, E)
        if i % 20 == 0:
            print(f"  [{tag}] {len(feats) * batch_size} frames ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(feats), torch.cat(states)


def ridge(x, y, lam: float):
    """Closed-form ridge with an intercept, fitted in float64 for conditioning."""
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1).double()
    y = y.double()
    a = x.T @ x + lam * torch.eye(x.shape[1], dtype=torch.float64)
    return torch.linalg.solve(a, x.T @ y)


def apply_ridge(w, x):
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1).double()
    return (x @ w).float()


def fit_linear(xtr, ytr, xva, yva, steps, lr, device):
    """Full-patch linear readout: too wide for a closed-form solve, so fit it with Adam."""
    n, d = xtr.shape[0], xtr.shape[1] * xtr.shape[2]
    mu, sd = ytr.mean(0), ytr.std(0).clamp_min(1e-6)
    layer = torch.nn.Linear(d, ytr.shape[1]).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=lr, weight_decay=1e-4)
    xtr_f, ytr_n = xtr.reshape(n, d), ((ytr - mu) / sd)
    best = None
    for step in range(1, steps + 1):
        sel = torch.randint(0, n, (128,))
        pred = layer(xtr_f[sel].to(device).float())
        loss = torch.nn.functional.mse_loss(pred, ytr_n[sel].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                out = torch.cat([
                    layer(xva[i:i + 64].reshape(-1, d).to(device).float()).cpu()
                    for i in range(0, xva.shape[0], 64)
                ]) * sd + mu
            mae = (out - yva).abs().mean().item()
            best = mae if best is None else min(best, mae)
            print(f"  [all_patches] step {step} heldout_mae={mae:.5f} best={best:.5f}", flush=True)
    return best, out


def report(pred, y, names):
    err = (pred - y).abs()
    return {"mae": round(err.mean().item(), 6),
            "per_joint_mae": {n: round(v, 5) for n, v in zip(names, err.mean(0).tolist())}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--heldout-root", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--linear-steps", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from lerobot.utils.constants import ACTION

    policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, args.device)

    tr_repo = f"{args.train_root.parent.name}/{args.train_root.name}"
    va_repo = f"{args.heldout_root.parent.name}/{args.heldout_root.name}"
    tr_ds, meta, _ = build_dataset(args.train_root, tr_repo, cfg, None)
    exclude = set(episode_action_hashes(args.train_root).values())
    va_ds, _, _ = build_dataset(args.heldout_root, va_repo, cfg, exclude, False)
    names = meta.features[ACTION].get("names")

    tr_idx = list(range(0, len(tr_ds), max(len(tr_ds) // args.n_train, 1)))[: args.n_train]
    va_idx = list(range(0, len(va_ds), max(len(va_ds) // args.n_val, 1)))[: args.n_val]
    xtr, ytr = extract(policy, preprocessor, cfg, tr_ds, tr_idx, args.batch_size, args.device, "train")
    xva, yva = extract(policy, preprocessor, cfg, va_ds, va_idx, args.batch_size, args.device, "val")
    print(f"features {tuple(xtr.shape)} / {tuple(xva.shape)}", flush=True)

    res = {"checkpoint": str(args.checkpoint), "train": tr_repo, "heldout": va_repo,
           "n_train": len(tr_idx), "n_val": len(va_idx),
           "vision_encoder": cfg.vision_encoder, "tokens_per_frame": xtr.shape[1],
           "readouts": {}}

    res["readouts"]["train_mean"] = report(ytr.mean(0, keepdim=True).expand_as(yva), yva, names)

    mtr, mva = xtr.float().mean(1), xva.float().mean(1)
    best = None
    for lam in (1e-2, 1e0, 1e2, 1e4):
        w = ridge(mtr, ytr, lam)
        r = report(apply_ridge(w, mva), yva, names)
        r["lambda"] = lam
        print(f"  [mean_pool] lambda={lam:g} heldout_mae={r['mae']:.5f}", flush=True)
        if best is None or r["mae"] < best["mae"]:
            best = r
    res["readouts"]["mean_pool_ridge"] = best

    mae, pred = fit_linear(xtr, ytr, xva, yva, args.linear_steps, 1e-4, args.device)
    r = report(pred, yva, names)
    r["best_mae_over_training"] = round(mae, 6)
    res["readouts"]["all_patches_linear"] = r

    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v["mae"] for k, v in res["readouts"].items()}, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
