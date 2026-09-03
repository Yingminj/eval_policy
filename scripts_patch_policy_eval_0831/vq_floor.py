#!/usr/bin/env python
"""The accuracy ceiling the `vqbet` head cannot cross, whatever the GPT learns.

VQ-BeT emits an action chunk as `decode(code) + offset`.  The code is one of
`vqvae_n_embed ** 2 = 256` combinations (`ResidualVQ`, `vqvae_num_layers` is hard-wired to
2 in `modeling_vqbet.py:779`), chosen by a 256-way classifier.  So before asking how well
the policy *picks* a code, ask what the codebook can *represent*: push the ground-truth
chunks of the eval set through the frozen encoder/quantiser/decoder and measure what comes
back.  That reconstruction error is a floor on the code term -- the learned offset head can
claw some of it back, which is why the deployed policy is not simply worse than this number,
but nothing about training the GPT longer moves the codebook.

Also reports how much of the codebook is alive: a 256-way vocabulary that only ever emits
a handful of combinations is a much smaller model than the config suggests.

    /opt/robot-platform/train-venv/bin/python vq_floor.py --checkpoint ... --out vq_floor.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_chunk_eval import build_dataset, episode_action_hashes, load_policy  # noqa: E402


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from lerobot.utils.constants import ACTION

    policy, preprocessor, postprocessor, cfg = load_policy(args.checkpoint, "cuda")
    assert cfg.action_head == "vqbet", f"no VQ-VAE in a {cfg.action_head} head"
    vq = policy.model.head.vqvae_model
    start = cfg.n_obs_steps - 1

    exclude = set(episode_action_hashes(args.train_root).values()) if args.train_root else None
    repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
    ds, meta, _ = build_dataset(args.dataset_root, repo_id, cfg, exclude)
    idx = list(range(0, len(ds), args.stride))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    combos, n, sum_raw, sum_norm = Counter(), 0, 0.0, 0.0
    for batch in loader:
        for cam in meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0
        gt_raw = batch[ACTION][:, start:].clone().float()          # (B, chunk, A) raw units
        proc = preprocessor(batch)
        gt_norm = proc[ACTION][:, start:].float()                  # normalised, what the VQ sees

        _, (enc_loss, _, vq_code, _) = vq.vqvae_forward(gt_norm)
        latent = vq.get_embeddings_from_code(vq_code)
        recon = vq.get_action_from_latent(latent).view(gt_norm.shape)
        recon_raw = postprocessor(recon).float().cpu()

        b = gt_raw.shape[0]
        sum_norm += float(enc_loss) * b
        sum_raw += (recon_raw - gt_raw).abs().mean().item() * b
        for row in vq_code.view(b, -1).cpu().tolist():
            combos[tuple(row)] += 1
        n += b

    top = combos.most_common(10)
    res = {
        "checkpoint": str(args.checkpoint),
        "dataset": repo_id,
        "anchors": n,
        "codebook_size": cfg.vqvae_n_embed,
        "rvq_layers": vq.vqvae_num_layers,
        "possible_combinations": cfg.vqvae_n_embed ** vq.vqvae_num_layers,
        "used_combinations": len(combos),
        "top10_share": round(sum(c for _, c in top) / n, 4),
        "top10": [{"code": list(k), "share": round(c / n, 4)} for k, c in top],
        "reconstruction_mae_normalised": round(sum_norm / n, 6),
        "reconstruction_mae_raw": round(sum_raw / n, 6),
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
