#!/usr/bin/env python
"""Short controlled training runs that isolate *why* ACT-DiT's observation encoder dies.

`probe_encoder_collapse.py` shows the symptom: by 50k steps the last encoder layer's
output LayerNorm gain has fallen from 1.0 to 0.49 and the encoder output no longer
responds to the image at all, while plain ACT trained on the same data keeps both.  Two
things differ between the two policies and either could explain it:

  * `ACTDiTConfig` raises `optimizer_lr` from ACT's 1e-5 to 1e-4, and the ACT training
    preset has no LR scheduler, so a post-norm (`pre_norm=False`) encoder takes a 10x
    larger step with no warmup;
  * `act_dit` adds the adaLN road, which carries the robot state into every decoder layer
    multiplicatively.  If that road alone fits the training set, shrinking the encoder's
    output gain is a free loss reduction, and once the gain is small no gradient returns
    to the cameras.

Each arm below removes exactly one of those and trains for `--steps` on the real dataset
with the real loss, recording the encoder's health every `--probe-every` steps.  The arm
whose encoder survives names the cause.

Arms
----
  shipped          the configuration that produced the deployed checkpoint
  lowlr            `optimizer_lr` 1e-5 (ACT's), everything else shipped
  warmup           1e-4 with a linear warmup over `--warmup-steps`
  no_state_adaln   adaLN carries the flow timestep only; the robot state reaches the
                   decoder solely through the encoder token, exactly as in plain ACT

`no_state_adaln` is applied by monkeypatch rather than a config flag on purpose: this
script has to run against the unmodified checkpointed code so that `shipped` really is
what shipped.

Usage
-----
    PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src python train_ablation.py \
        --arm shipped --steps 6000 \
        --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361 \
        --out shipped.json

Self-check:  python train_ablation.py --selftest
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ARMS = ("shipped", "lowlr", "warmup", "no_state_adaln")


def apply_arm(arm: str, cfg) -> None:
    """Mutate `cfg` (and, for `no_state_adaln`, the policy module) for one arm."""
    if arm == "lowlr":
        cfg.optimizer_lr = 1e-5
    elif arm == "no_state_adaln":
        from lerobot.policies.act_dit import modeling_act_dit as mod

        # The decoder layer sizes its adaLN Linear from this, so it has to be patched
        # before the policy is constructed.
        mod._static_cond_dim = lambda c: 0
        mod.ACTDiT.encode_conditioning = lambda self, batch: (
            *self.encode_observations(batch),
            None,
        )
    elif arm not in ("shipped", "warmup"):
        raise ValueError(f"unknown arm {arm!r}")


@torch.no_grad()
def encoder_probe(policy, batch, batch_alt) -> dict:
    """mean|gamma| of the last encoder layer, and how much its output moves with the image."""
    model = policy.model
    a, _ = model.encode_observations(batch)
    b, _ = model.encode_observations(batch_alt)
    return {
        "gamma_last": model.encoder.layers[-1].norm2.weight.abs().mean().item(),
        "image_sensitivity": ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-12)).item(),
        "signal": a.abs().mean().item(),
    }


def train(arm: str, dataset_root: Path, steps: int, batch_size: int, probe_every: int,
          warmup_steps: int, seed: int, device: str, num_workers: int) -> dict:
    from lerobot.configs.policies import PreTrainedConfig  # noqa: F401  (registers subclasses)
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act_dit.configuration_act_dit import ACTDiTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import OBS_IMAGES

    torch.manual_seed(seed)
    repo_id = f"{dataset_root.parent.name}/{dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)

    cfg = ACTDiTConfig()
    cfg.device = device  # `make_policy` fills input/output_features from `meta`
    apply_arm(arm, cfg)

    ds = LeRobotDataset(repo_id=repo_id, root=dataset_root,
                        delta_timestamps=resolve_delta_timestamps(cfg, meta))
    policy = make_policy(cfg, ds_meta=meta)
    pre, _ = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy.train()

    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr,
                            weight_decay=cfg.optimizer_weight_decay)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True,
                                         pin_memory=device.startswith("cuda"))

    history, running, t0 = [], None, time.time()
    probe_batch = probe_alt = None
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = pre(batch)

        if arm == "warmup":
            scale = min(1.0, step / max(warmup_steps, 1))
            for g, base in zip(opt.param_groups, (cfg.optimizer_lr, cfg.optimizer_lr_backbone)):
                g["lr"] = base * scale

        loss, _ = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        policy.update()  # EMA hook; a no-op unless use_ema

        running = loss.item() if running is None else 0.99 * running + 0.01 * loss.item()

        if step % probe_every == 0 or step == 1:
            prepared = policy._prepare_batch(batch)
            if probe_batch is None:
                # Frozen probe inputs: the encoder health trace has to be measured on the
                # same two observations at every step or the trace mixes in batch variation.
                probe_batch = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                               for k, v in prepared.items() if k != OBS_IMAGES}
                probe_batch[OBS_IMAGES] = [i.detach().clone() for i in prepared[OBS_IMAGES]]
                roll = torch.roll(torch.arange(prepared[OBS_IMAGES][0].shape[0], device=device), 1)
                probe_alt = {**probe_batch,
                             OBS_IMAGES: [i[roll] for i in probe_batch[OBS_IMAGES]]}
            policy.eval()
            row = {"step": step, "loss": round(running, 5),
                   **{k: round(v, 6) for k, v in encoder_probe(policy, probe_batch, probe_alt).items()}}
            policy.train()
            history.append(row)
            print(f"[{arm}] {row}  ({time.time() - t0:.0f}s)", flush=True)

    return {"arm": arm, "steps": steps, "batch_size": batch_size, "lr": cfg.optimizer_lr,
            "seed": seed, "dataset": repo_id, "seconds": round(time.time() - t0, 1),
            "history": history}


def selftest() -> None:
    """`apply_arm` must actually change what it claims to, and nothing else."""
    from types import SimpleNamespace

    c = SimpleNamespace(optimizer_lr=1e-4)
    apply_arm("shipped", c)
    assert c.optimizer_lr == 1e-4
    apply_arm("warmup", c)
    assert c.optimizer_lr == 1e-4, "warmup must not change the peak LR"
    apply_arm("lowlr", c)
    assert c.optimizer_lr == 1e-5, c.optimizer_lr
    try:
        apply_arm("nope", c)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arm was accepted")

    # encoder_probe's sensitivity must be 0 for an encoder that ignores its input, and
    # positive for one that does not.
    class Enc(torch.nn.Module):
        def __init__(self, live):
            super().__init__()
            self.live = live
            self.layers = [SimpleNamespace(norm2=SimpleNamespace(weight=torch.full((4,), 0.5)))]

    class M(torch.nn.Module):
        def __init__(self, live):
            super().__init__()
            self.encoder = Enc(live)
            self.live = live

        def encode_observations(self, b):
            return (b["x"] if self.live else torch.ones_like(b["x"])), None

    for live, expect_zero in ((True, False), (False, True)):
        p = SimpleNamespace(model=M(live))
        r = encoder_probe(p, {"x": torch.ones(3, 4)}, {"x": torch.full((3, 4), 3.0)})
        assert r["gamma_last"] == 0.5, r
        assert (r["image_sensitivity"] == 0.0) == expect_zero, r
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.arm or not a.dataset_root:
        p.error("--arm and --dataset-root are required")
    selftest()
    r = train(a.arm, a.dataset_root, a.steps, a.batch_size, a.probe_every,
              a.warmup_steps, a.seed, a.device, a.num_workers)
    if a.out:
        a.out.write_text(json.dumps(r, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
