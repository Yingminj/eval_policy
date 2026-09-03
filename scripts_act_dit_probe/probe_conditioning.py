#!/usr/bin/env python
"""Where does an ACT-DiT chunk actually come from? Attribution over the three
conditioning routes the architecture provides.

`act_dit` feeds the observation to its decoder by three different roads:

  1. camera tokens  -> transformer encoder -> decoder cross-attention   (~900 tokens)
  2. robot state    -> transformer encoder -> decoder cross-attention   (1 token)
  3. robot state    -> `static_cond`       -> adaLN-Zero, every layer   (512-d, multiplicative)

Road 3 is what `act_dit` adds on top of plain ACT (which has only 1 and 2), and it is
privileged: adaLN shift/scale/gate multiply *every* self-attention and FFN branch in
*every* decoder layer, whereas the visual tokens only ever arrive as one additive
residual per layer.  This script measures how much the emitted chunk moves when each
road is fed a different frame's observation, so the split between them is a number
rather than an argument.

All comparisons share one fixed initial noise tensor (`--noise-seed`), because the
flow-matching sample is stochastic and a fresh draw per condition would swamp the
effect being measured.  Deltas are reported both in radians and as a fraction of the
*between-frame spread* -- the mean pairwise distance between chunks of different
anchors -- which is the only scale on which "moved a lot" means anything.

Also reported:
  * `time_vs_state_norm` - the L2 norms of the two halves of the adaLN conditioning
    vector, `[time_mlp(t) | state]`.  These are concatenated and fed to one Linear, so
    a large imbalance means the noise level is a rounding error next to the state.
  * `dfdt` - how far the denoiser's output moves when only `t` changes.  A flow model
    whose velocity field is nearly constant in `t` is not really denoising.

Usage
-----
    PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src python probe_conditioning.py \
        --checkpoint /mnt/.../checkpoints/200000/pretrained_model \
        --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361 \
        --n-anchors 64 --out probe.json

Self-check:  python probe_conditioning.py --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


# --------------------------------------------------------------------------------------
def euler_sample(model, x0: torch.Tensor, cond, n_steps: int) -> torch.Tensor:
    """`FlowMatchingObjective._euler_integrate` with the initial noise supplied.

    Copied rather than called because the objective draws its own `randn`, and every
    condition in this probe has to start from the *same* point of the noise space.
    """
    x = x0
    grid = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
    for i in range(n_steps):
        dt = (grid[i + 1] - grid[i]).item()
        t = torch.full((x.shape[0],), grid[i].item(), dtype=x.dtype, device=x.device)
        x = x + dt * model(x, t, conditioning_vec=cond)
    return x


def ddim_sample(objective, model, x0: torch.Tensor, cond) -> torch.Tensor:
    """`DiffusionObjective.conditional_sample` with the initial noise supplied.

    Same reason as `euler_sample`: the objective draws its own `randn`, and every condition
    in this probe has to start from the same point of the noise space. The scheduler is the
    checkpoint's own, so `num_inference_steps`, the beta schedule and `prediction_type` all
    come from the trained config.
    """
    sample = x0
    sched = objective.noise_scheduler
    sched.set_timesteps(objective.num_inference_steps)
    for t in sched.timesteps:
        out = model(sample, torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                    conditioning_vec=cond)
        sample = sched.step(out, t, sample).prev_sample
    return sample


def make_sampler(policy, cfg):
    """One `(x0, cond) -> chunk` callable, whichever objective the checkpoint was trained with."""
    if cfg.objective == "flow_matching":
        return lambda x0, cond: euler_sample(model_of(policy), x0, cond, cfg.num_integration_steps)
    return lambda x0, cond: ddim_sample(policy.objective, model_of(policy), x0, cond)


def model_of(policy):
    return policy.model


def timestep_probe_values(policy, cfg, b: int, device: str):
    """Four timesteps spanning the objective's own noise axis, in its own units.

    Flow matching integrates t in [0, 1]; diffusion indexes integer training timesteps, and
    the scheduler's inference grid is the only place the model was ever evaluated.
    """
    if cfg.objective == "flow_matching":
        return [torch.full((b,), v, device=device) for v in (0.05, 0.35, 0.65, 0.95)]
    sched = policy.objective.noise_scheduler
    sched.set_timesteps(policy.objective.num_inference_steps)
    ts = sched.timesteps
    picks = [ts[int(round(f * (len(ts) - 1)))] for f in (0.95, 0.65, 0.35, 0.05)]
    return [torch.full((b,), int(t), dtype=torch.long, device=device) for t in picks]


def conditioning(model, batch: dict, state_token: torch.Tensor, state_adaln: torch.Tensor):
    """`ACTDiT.encode_conditioning` with the two state roads fed separately.

    Only valid for `use_cross_attention=True` (the shipped default), where `static_cond`
    is exactly the projected robot state and nothing else; the ablation arm also concatenates
    a pooled encoder output, which this split would silently drop.
    """
    if not model.config.use_cross_attention:
        raise ValueError("split conditioning assumes use_cross_attention=True")
    enc_out, enc_pos = model.encode_observations({**batch, OBS_STATE: state_token})
    return enc_out, enc_pos, model.encoder_robot_state_input_proj(state_adaln)


def pairwise_spread(chunks: torch.Tensor) -> float:
    """Mean L1 distance between the chunks of *different* anchors.

    The natural scale of "this chunk is different from that chunk" for this dataset and
    this policy. A perturbation that moves a chunk by 10% of this has barely moved it.
    """
    n = chunks.shape[0]
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += (chunks[i] - chunks[j]).abs().mean().item()
            count += 1
    return total / max(count, 1)


# --------------------------------------------------------------------------------------
@torch.no_grad()
def probe(checkpoint: Path, dataset_root: Path, n_anchors: int, stride: int,
          device: str, noise_seed: int) -> dict:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path, cfg.device = str(checkpoint), device
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device)
    policy.eval()  # also swaps the EMA shadow in, which is what deployment samples from
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    model = policy.model

    repo_id = f"{dataset_root.parent.name}/{dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    ds = LeRobotDataset(repo_id=repo_id, root=dataset_root,
                        delta_timestamps=resolve_delta_timestamps(cfg, meta))
    idx = list(range(0, len(ds), stride))[:n_anchors]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx),
                                         batch_size=len(idx), num_workers=4)
    batch = next(iter(loader))
    for cam in meta.camera_keys:
        if batch[cam].dtype == torch.uint8:
            batch[cam] = batch[cam].to(torch.float32) / 255.0
    batch = policy._prepare_batch(pre(batch))  # adds the OBS_IMAGES list the encoder reads

    b = batch[OBS_STATE].shape[0]
    roll = torch.roll(torch.arange(b, device=device), 1)  # "another frame's" observation
    state = batch[OBS_STATE]

    torch.manual_seed(noise_seed)
    x0 = torch.randn(b, cfg.chunk_size, cfg.action_feature.shape[0], device=device)

    sampler = make_sampler(policy, cfg)

    def run(bt: dict, s_token: torch.Tensor, s_adaln: torch.Tensor) -> torch.Tensor:
        chunk = sampler(x0, conditioning(model, bt, s_token, s_adaln))
        return post(chunk).float().cpu()  # raw joint units (rad)

    # `encode_observations` reads OBS_IMAGES (the list `_prepare_batch` built), not the
    # per-camera keys, so the swap has to be applied there or it is silently a no-op.
    swapped_imgs = {**batch, OBS_IMAGES: [img[roll] for img in batch[OBS_IMAGES]]}
    conds = {
        "baseline":          run(batch, state, state),
        # every camera replaced by another frame's, state untouched
        "images_swapped":    run(swapped_imgs, state, state),
        # state replaced on both roads, images untouched
        "state_swapped":     run(batch, state[roll], state[roll]),
        # state replaced on the adaLN road only (the road act_dit adds over ACT)
        "state_swap_adaln":  run(batch, state, state[roll]),
        # state replaced on the encoder-token road only (the road plain ACT has)
        "state_swap_token":  run(batch, state[roll], state),
        # state set to the dataset mean (0 after normalisation) on both roads
        "state_zeroed":      run(batch, torch.zeros_like(state), torch.zeros_like(state)),
    }

    base = conds["baseline"]
    spread = pairwise_spread(base)
    deltas = {k: (v - base).abs().mean().item() for k, v in conds.items() if k != "baseline"}

    # --- how loud is the timestep next to the state, inside the same adaLN vector? ---
    t_probe = timestep_probe_values(policy, cfg, b, device)
    t_emb = model.time_mlp(t_probe[len(t_probe) // 2])
    s_emb = model.encoder_robot_state_input_proj(state)
    lin = model.decoder.layers[0].adaln[-1]  # SiLU -> Linear
    n_t = t_emb.shape[-1]

    # --- does the velocity field depend on t at all? ---
    cond0 = conditioning(model, batch, state, state)
    xs = x0 * 0.5 + base.to(device) * 0.5  # a plausible mid-trajectory point
    outs = [model(xs, tv, conditioning_vec=cond0) for tv in t_probe]
    stacked = torch.stack(outs)
    dfdt = (stacked - stacked.mean(0, keepdim=True)).abs().mean().item()

    return {
        "checkpoint": str(checkpoint),
        "objective": cfg.objective,
        "dataset": repo_id,
        "n_anchors": b,
        "noise_seed": noise_seed,
        "between_frame_spread_rad": spread,
        "delta_rad": {k: round(v, 6) for k, v in deltas.items()},
        "delta_frac_of_spread": {k: round(v / max(spread, 1e-9), 4) for k, v in deltas.items()},
        "adaln_conditioning": {
            "time_norm": round(t_emb.norm(dim=-1).mean().item(), 4),
            "state_norm": round(s_emb.norm(dim=-1).mean().item(), 4),
            "weight_col_norm_time": round(lin.weight[:, :n_t].norm().item(), 4),
            "weight_col_norm_state": round(lin.weight[:, n_t:].norm().item(), 4),
            "contribution_time": round((t_emb @ lin.weight[:, :n_t].T).norm(dim=-1).mean().item(), 4),
            "contribution_state": round((s_emb @ lin.weight[:, n_t:].T).norm(dim=-1).mean().item(), 4),
        },
        "velocity_field": {
            "spread_over_t": round(dfdt, 6),
            "magnitude": round(stacked.abs().mean().item(), 6),
        },
    }


# --------------------------------------------------------------------------------------
def selftest() -> None:
    """`euler_sample` must integrate a known field exactly, and `pairwise_spread` must
    match a hand-computed case. Everything else here is measurement plumbing."""
    class ConstField(torch.nn.Module):
        def forward(self, x, t, conditioning_vec=None):
            return torch.ones_like(x) * 2.0

    x0 = torch.zeros(2, 3, 4)
    out = euler_sample(ConstField(), x0, None, 10)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-5), out.mean()

    class LinearInT(torch.nn.Module):
        def forward(self, x, t, conditioning_vec=None):
            return t.view(-1, 1, 1).expand_as(x).clone()

    # integral of t dt from 0 to 1 = 0.5; left-endpoint Euler with 10 steps gives 0.45
    out = euler_sample(LinearInT(), torch.zeros(1, 2, 2), None, 10)
    assert abs(out.mean().item() - 0.45) < 1e-5, out.mean()

    # ddim_sample must honour the supplied x0 and be deterministic given it
    class Eps(torch.nn.Module):
        def forward(self, x, t, conditioning_vec=None):
            return torch.zeros_like(x)

    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    class Obj:
        noise_scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
        num_inference_steps = 10

    x0 = torch.randn(2, 3, 4)
    a1 = ddim_sample(Obj(), Eps(), x0, None)
    a2 = ddim_sample(Obj(), Eps(), x0, None)
    assert torch.equal(a1, a2), "ddim_sample not deterministic for a fixed x0"
    b1 = ddim_sample(Obj(), Eps(), x0 * 2.0, None)
    assert not torch.allclose(a1, b1), "ddim_sample ignored the supplied initial noise"

    c = torch.tensor([[[0.0]], [[1.0]], [[4.0]]])  # pairwise |.|: 1, 4, 3 -> mean 8/3
    assert abs(pairwise_spread(c) - 8.0 / 3.0) < 1e-9, pairwise_spread(c)
    assert pairwise_spread(c[:1]) == 0.0
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--n-anchors", type=int, default=64)
    p.add_argument("--stride", type=int, default=997, help="prime, to spread anchors over episodes")
    p.add_argument("--device", default="cuda")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.checkpoint or not a.dataset_root:
        p.error("--checkpoint and --dataset-root are required")
    selftest()
    r = probe(a.checkpoint, a.dataset_root, a.n_anchors, a.stride, a.device, a.noise_seed)
    print(json.dumps(r, indent=2))
    if a.out:
        a.out.write_text(json.dumps(r, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
