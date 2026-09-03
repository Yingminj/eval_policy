#!/usr/bin/env python
"""How much of the denoiser's cross-attention actually lands on the state token?

Zero-training check of the report's hypothesis 7.3-1 ("the state token is drowned in 768 patch
tokens"). The denoiser's memory is `[time] + n_obs_steps * tokens_per_frame` slots, and with
`use_robot_state=True` exactly one slot per frame is the state. Two numbers per checkpoint:

  attn_mass_state  -- softmax mass the decoder's cross-attention puts on the state slots,
                      averaged over layers/heads/action-positions. Fair share = n_state/n_memory.
  norm_ratio       -- ||state slot|| / mean ||patch slot|| in memory space (post `cond_obs_emb`
                      and post the memory MLP). Tests hypothesis 7.3-3 (scale mismatch) with the
                      same forward pass.

A mass at or below fair share means the token is not being sought out; a norm ratio far below 1
means it could not compete even if it were.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402


def capture_attn(mha, store):
    """nn.TransformerDecoderLayer calls multihead_attn(need_weights=False); force it True."""
    inner = mha.forward

    def forward(*a, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = False
        out, w = inner(*a, **kw)
        store.append(w.detach())          # (B, heads, horizon, n_memory)
        return out, None

    mha.forward = forward


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--n-anchors", type=int, default=16)
    p.add_argument("--stride", type=int, default=997)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from lerobot.utils.constants import OBS_IMAGES

    policy, preprocessor, _, cfg = load_policy(args.checkpoint, args.device)
    if not cfg.use_robot_state:
        sys.exit("checkpoint has use_robot_state=False -- nothing to measure")
    if cfg.action_head != "diffusion":
        sys.exit(f"only the diffusion head is instrumented, got {cfg.action_head}")

    model = policy.model
    P = model.tokens_per_frame
    T = cfg.n_obs_steps
    n_mem = 1 + T * P
    # memory = [time] + per frame [768 patch tokens ... 1 state token]; state is last in its block.
    state_slots = [1 + t * P + (P - 1) for t in range(T)]

    store = []
    for layer in model.head.decoder.layers:
        capture_attn(layer.multihead_attn, store)

    exclude = set(episode_action_hashes(args.train_root).values()) if args.train_root else None
    repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
    ds, meta, _ = build_dataset(args.dataset_root, repo_id, cfg, exclude, False)
    idx = list(range(0, len(ds), args.stride))[: args.n_anchors]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=4, shuffle=False, num_workers=4
    )

    mass_state = mass_time = 0.0
    norm_state = norm_patch = 0.0
    n = 0
    for batch in loader:
        for cam in meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
        proc = preprocessor(batch)
        proc[OBS_IMAGES] = torch.stack([proc[k] for k in cfg.image_features], dim=-4)

        store.clear()
        with torch.no_grad():
            tokens = model.encode_observations(proc)
            cond = tokens.flatten(1, 2)
            memory = model.head.encoder(
                torch.cat(
                    [
                        model.head.time_emb(torch.zeros(cond.shape[0], device=cond.device)).unsqueeze(1),
                        model.head.cond_obs_emb(cond),
                    ],
                    dim=1,
                )
                + model.head.cond_pos_emb
            )
            model.predict(proc)

        norms = memory.norm(dim=-1)                                   # (B, n_mem)
        keep = torch.ones(n_mem, dtype=torch.bool)
        keep[state_slots] = False
        keep[0] = False
        b = memory.shape[0]
        norm_state += norms[:, state_slots].mean().item() * b
        norm_patch += norms[:, keep].mean().item() * b

        # one entry per (layer, denoising step); all share the same memory layout
        w = torch.stack(store).float()                                # (L*steps, B, heads, H, n_mem)
        mass_state += w[..., state_slots].sum(-1).mean().item() * b
        mass_time += w[..., 0].mean().item() * b
        n += b

    res = {
        "checkpoint": str(args.checkpoint),
        "n_obs_steps": T,
        "tokens_per_frame": P,
        "n_memory_slots": n_mem,
        "n_state_slots": len(state_slots),
        "fair_share": len(state_slots) / n_mem,
        "attn_mass_state": mass_state / n,
        "attn_mass_time_token": mass_time / n,
        "norm_state_slot": norm_state / n,
        "norm_patch_slot": norm_patch / n,
        "norm_ratio_state_over_patch": (norm_state / n) / (norm_patch / n),
        "anchors": n,
    }
    res["attn_mass_over_fair_share"] = res["attn_mass_state"] / res["fair_share"]
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
