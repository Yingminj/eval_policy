#!/usr/bin/env python
"""Is the observation encoder still alive? A per-checkpoint health check for ACT-family policies.

ACT's encoder is post-norm: each layer ends in `x = norm2(x + ffn(...))`, so the last
layer's `norm2.weight` (gamma) is a per-channel *global gain* on everything the decoder's
cross-attention can ever see.  That makes it a one-vector off switch for the whole
observation path.

Plain ACT cannot flip that switch: its decoder input is a zero tensor, so cross-attention
is the only road from observation to action and killing it costs all the loss.  `act_dit`
*can*: it adds an adaLN road carrying the robot state (and the flow timestep) straight
into every decoder layer, multiplicatively.  If that road is enough to fit the training
set, gradient descent is free to shrink the encoder's output gain toward zero -- and once
it does, no gradient flows back to the cameras, so the collapse is self-locking.

This script measures three things per checkpoint and prints them side by side:

  `gamma`        - mean |gamma| of every encoder layer's output LayerNorm. 1.0 is init.
  `signal`       - abs-mean of the encoder output that cross-attention receives, traced
                   layer by layer through a real forward pass, so an attenuation shows up
                   at the layer where it happens.
  `image_sens`   - relative change in the encoder output when the images are replaced by
                   a different frame's.  This is the number that matters: an encoder whose
                   output does not move when the scene changes is not an encoder.

Run it on every checkpoint of a run to see *when* the encoder died.

Usage
-----
    PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src python probe_encoder_collapse.py \
        --checkpoint /mnt/.../checkpoints/050000/pretrained_model \
        --checkpoint /mnt/.../checkpoints/200000/pretrained_model \
        --out collapse.json

Self-check:  python probe_encoder_collapse.py --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def encoder_health(policy, cfg, images: torch.Tensor, images_alt: torch.Tensor) -> dict:
    """Trace the encoder on two different image batches; report gain, signal and sensitivity.

    `images`/`images_alt` are (B, 3, H, W) float tensors already scaled to [0, 1] and
    normalised the way the checkpoint's preprocessor would; the absolute level does not
    matter here because every quantity reported is either a weight statistic or a ratio.
    """
    import einops

    m = policy.model
    dev = images.device
    b = images.shape[0]

    def encode(img: torch.Tensor) -> list[torch.Tensor]:
        feat = m.backbone(img)["feature_map"]
        tok = einops.rearrange(m.encoder_img_feat_input_proj(feat), "b c h w -> (h w) b c")
        pos_img = einops.rearrange(
            m.encoder_cam_feat_pos_embed(feat).to(feat.dtype), "b c h w -> (h w) b c"
        )[:, :1]
        head = torch.stack(
            [
                m.encoder_latent_input_proj(torch.zeros(b, cfg.latent_dim, device=dev)),
                m.encoder_robot_state_input_proj(
                    torch.zeros(b, cfg.robot_state_feature.shape[0], device=dev)
                ),
            ]
        )
        x = torch.cat([head, tok], dim=0)
        pos = torch.cat([m.encoder_1d_feature_pos_embed.weight.unsqueeze(1), pos_img], dim=0)
        outs = []
        for layer in m.encoder.layers:
            x = layer(x, pos_embed=pos)
            outs.append(x)
        return outs

    a, c = encode(images), encode(images_alt)
    gammas = [layer.norm2.weight for layer in m.encoder.layers]
    return {
        "gamma_mean_abs": [round(g.abs().mean().item(), 5) for g in gammas],
        "gamma_frac_below_0.05": [round((g.abs() < 0.05).float().mean().item(), 4) for g in gammas],
        "signal_abs_mean": [round(x.abs().mean().item(), 6) for x in a],
        # relative, so it is comparable between a healthy and an attenuated encoder
        "image_sensitivity": [
            round(((x - y).abs().mean() / x.abs().mean().clamp_min(1e-12)).item(), 6)
            for x, y in zip(a, c)
        ],
    }


def load(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path, cfg.device = str(checkpoint), device
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device)
    policy.eval()
    return policy, cfg


def run(checkpoints: list[Path], device: str, seed: int) -> dict:
    # Two fixed random image batches. Real frames would be more realistic, but the question
    # here is only "does the output move when the input moves", and noise answers it without
    # dragging a dataset (and its video decoder) into a weight-level check.
    g = torch.Generator(device="cpu").manual_seed(seed)
    img_a = torch.rand(2, 3, 480, 640, generator=g).to(device)
    img_b = torch.rand(2, 3, 480, 640, generator=g).to(device)

    out = {}
    for ck in checkpoints:
        policy, cfg = load(ck, device)
        out[str(ck)] = {"policy_type": cfg.type, **encoder_health(policy, cfg, img_a, img_b)}
        del policy
        torch.cuda.empty_cache()
    return out


def report(results: dict) -> None:
    for name, r in results.items():
        print(f"\n{name}  [{r['policy_type']}]")
        n = len(r["gamma_mean_abs"])
        print("  layer   mean|gamma|   frac<0.05   signal   img-sensitivity")
        for i in range(n):
            print(
                f"  enc{i}      {r['gamma_mean_abs'][i]:8.4f}     {r['gamma_frac_below_0.05'][i]:6.3f}"
                f"   {r['signal_abs_mean'][i]:8.5f}   {r['image_sensitivity'][i]:.6f}"
            )


def selftest() -> None:
    """A hand-built two-layer post-norm encoder: one healthy layer, one with gamma zeroed.

    Checks that `encoder_health` localises the attenuation to the layer that causes it and
    reports that layer as image-insensitive, which is the whole claim the script makes.
    """
    from types import SimpleNamespace

    class Layer(torch.nn.Module):
        def __init__(self, gain: float):
            super().__init__()
            self.norm2 = torch.nn.LayerNorm(8)
            torch.nn.init.constant_(self.norm2.weight, gain)

        def forward(self, x, pos_embed=None):
            return self.norm2(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(layers=[Layer(1.0), Layer(0.0)])
            self.backbone = lambda img: {"feature_map": img.mean(dim=(2, 3))[:, :, None, None].expand(-1, 8, 3, 3)}
            self.encoder_img_feat_input_proj = torch.nn.Identity()
            self.encoder_cam_feat_pos_embed = lambda f: torch.zeros_like(f)
            self.encoder_latent_input_proj = torch.nn.Linear(4, 8)
            self.encoder_robot_state_input_proj = torch.nn.Linear(6, 8)
            self.encoder_1d_feature_pos_embed = torch.nn.Embedding(2, 8)

    policy = SimpleNamespace(model=Model())
    cfg = SimpleNamespace(latent_dim=4, robot_state_feature=SimpleNamespace(shape=(6,)))
    r = encoder_health(policy, cfg, torch.rand(2, 8, 4, 4), torch.rand(2, 8, 4, 4))

    assert r["gamma_mean_abs"] == [1.0, 0.0], r["gamma_mean_abs"]
    assert r["gamma_frac_below_0.05"] == [0.0, 1.0], r["gamma_frac_below_0.05"]
    # layer 1 zeroes everything, so its signal is 0 and its sensitivity is 0/0 -> 0
    assert r["signal_abs_mean"][1] == 0.0, r["signal_abs_mean"]
    assert r["image_sensitivity"][1] == 0.0, r["image_sensitivity"]
    # layer 0 must still react to a different image
    assert r["image_sensitivity"][0] > 1e-3, r["image_sensitivity"]
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.checkpoint:
        p.error("at least one --checkpoint is required")
    selftest()
    results = run(a.checkpoint, a.device, a.seed)
    report(results)
    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
