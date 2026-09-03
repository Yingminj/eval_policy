#!/usr/bin/env python
"""Short controlled `patch_policy` training runs, scored on held-out data.

Training loss is not a usable score for this corpus: `../scripts_act_eval_test/README.md`
shows an ACT checkpoint reaching 0.042 training loss while being only 1.21x better than
doing nothing on episodes it did not train on. So every arm here is scored the way the
deployed policy is judged -- open-loop chunk MAE on **held-out** episodes, against the
`hold_state` null baseline -- measured every `--val-every` steps on one fixed anchor set.

The base config is read from a real checkpoint directory rather than from
`PatchPolicyConfig()` defaults, so the `shipped` arm is genuinely what shipped; each arm
then applies `--override key=value` on top of it and nothing else changes.

Usage
-----
    python train_patch_ablation.py --name state --override use_robot_state=True \
        --steps 20000 --base-config <ckpt-dir> \
        --dataset-root .../batch_success_361 --heldout-root .../batch_3 \
        --train-root .../batch_success_361 --out state.json

Self-check:  python train_patch_ablation.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts_act_eval_test"))


def apply_state_broadcast() -> None:
    """Make the robot state unavoidable instead of one token in 3846.

    With `use_robot_state=True` the shipped code appends ONE projected state token per frame.
    Measured on the real mask, that token is 5 of 3846 memory entries (0.130%), reachable only
    if cross-attention learns to single it out -- the memory encoder is a 2-layer MLP, so the
    state cannot broadcast itself to the other tokens either. The `state` arm shows this is not
    learned in 20k steps.

    This arm keeps the appended token (so `tokens_per_frame` and the block-causal mask are
    unchanged and the comparison stays like-for-like) and ADDITIONALLY adds the projected state
    to every patch token of its own frame. Every memory entry now carries the pose, so using it
    costs the decoder no attention selectivity at all.

    Monkeypatched rather than added as a config flag so the other arms keep running the
    unmodified shipped code.
    """
    import einops
    from lerobot.policies.patch_policy import modeling_patch_policy as mod
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    def encode_observations(self, batch):
        batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]
        images = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
        with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
            patch_tokens = self.encoder(images)
        patch_tokens = einops.rearrange(
            patch_tokens, "(b s n) p e -> b s (n p) e",
            b=batch_size, s=n_obs_steps, n=self.num_images,
        )
        state_token = self.state_projector(batch[OBS_STATE]).unsqueeze(2)  # (b, s, 1, e)
        patch_tokens = patch_tokens + state_token                          # broadcast over tokens
        return torch.cat([patch_tokens, state_token], dim=2)

    mod.PatchPolicyModel.encode_observations = encode_observations


def apply_overrides(cfg, overrides: list[str]) -> dict:
    """`key=value` strings onto `cfg`, values parsed as Python literals.

    An unknown key is an error rather than a silent no-op: a typo'd override would
    otherwise produce an arm identical to `shipped` and be reported as a null result.
    """
    applied = {}
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        if not hasattr(cfg, key):
            raise ValueError(f"{type(cfg).__name__} has no field {key!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(cfg, key, value)
        applied[key] = value
    return applied


@torch.no_grad()
def validate(policy, preprocessor, postprocessor, loader, cfg, device, waypoint: int) -> dict:
    """Open-loop chunk MAE on held-out anchors, in raw joint units."""
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    start = cfg.n_obs_steps - 1
    was_training = policy.training
    policy.eval()
    tot = {"policy": 0.0, "hold_state": 0.0, "policy_first": 0.0, "hold_first": 0.0,
           "policy_exec": 0.0, "hold_exec": 0.0, "waypoint": 0.0, "waypoint_reanchored": 0.0,
           "waypoint_hold": 0.0}
    n = 0
    # The deploy path (`n_action_steps: 8`, `bridge_steps = min(40, 8)`) replaces the whole
    # emitted chunk with a Hermite from the measured pose to `chunk[-1]`, so `chunk[K-1]` is
    # the only number the model contributes. `waypoint` scores exactly that.
    exec_len = min(cfg.n_action_steps, cfg.action_chunk_size)
    wp = min(waypoint, cfg.action_chunk_size) - 1

    for batch in loader:
        # The postprocessor un-normalises onto the CPU, so the targets stay there too.
        gt = batch[ACTION][:, start:].clone().float()
        state = batch[OBS_STATE][:, -1].clone().float()
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        for key in cfg.image_features:
            if batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        proc = preprocessor(dict(batch))
        proc[OBS_IMAGES] = torch.stack([proc[k] for k in cfg.image_features], dim=-4)
        pred = postprocessor(policy.model.predict(proc)).float().cpu()

        b = gt.shape[0]
        hold = state.unsqueeze(1).expand_as(gt)
        tot["policy"] += (pred - gt).abs().mean().item() * b
        tot["hold_state"] += (hold - gt).abs().mean().item() * b
        tot["policy_first"] += (pred[:, 0] - gt[:, 0]).abs().mean().item() * b
        tot["hold_first"] += (hold[:, 0] - gt[:, 0]).abs().mean().item() * b
        tot["policy_exec"] += (pred[:, :exec_len] - gt[:, :exec_len]).abs().mean().item() * b
        tot["hold_exec"] += (hold[:, :exec_len] - gt[:, :exec_len]).abs().mean().item() * b
        tot["waypoint"] += (pred[:, wp] - gt[:, wp]).abs().mean().item() * b
        tot["waypoint_reanchored"] += (
            (pred[:, wp] - pred[:, 0] + state) - gt[:, wp]).abs().mean().item() * b
        tot["waypoint_hold"] += (state - gt[:, wp]).abs().mean().item() * b
        n += b

    if was_training:
        policy.train()
    out = {k: round(v / n, 6) for k, v in tot.items()}
    out["anchors"] = n
    out["gain_vs_null"] = round(out["hold_state"] / max(out["policy"], 1e-9), 3)
    out["exec_gain_vs_null"] = round(out["hold_exec"] / max(out["policy_exec"], 1e-9), 3)
    out["waypoint_gain_vs_null"] = round(out["waypoint_hold"] / max(out["waypoint"], 1e-9), 3)
    return out


def build(args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from offline_chunk_eval import build_dataset, episode_action_hashes

    cfg = PreTrainedConfig.from_pretrained(args.base_config)
    cfg.pretrained_path = None           # build fresh weights, not the trained ones
    cfg.device = args.device
    applied = apply_overrides(cfg, args.override)
    if args.state_broadcast:
        if not cfg.use_robot_state:
            raise SystemExit("--state-broadcast needs --override use_robot_state=True")
        apply_state_broadcast()
        applied["state_broadcast"] = True

    repo_id = f"{args.dataset_root.parent.name}/{args.dataset_root.name}"
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=args.dataset_root)
    ds = LeRobotDataset(repo_id=repo_id, root=args.dataset_root,
                        delta_timestamps=resolve_delta_timestamps(cfg, meta))

    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, dataset_stats=meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    exclude = set(episode_action_hashes(args.train_root).values()) if args.train_root else None
    val_repo = f"{args.heldout_root.parent.name}/{args.heldout_root.name}"
    val_ds, _, _ = build_dataset(args.heldout_root, val_repo, cfg, exclude, False)
    if val_ds is None:
        raise SystemExit(f"{val_repo} is entirely contained in the training set")
    # A prime stride spreads the anchors across episodes and phases of the task instead of
    # clustering them at the start of the first few episodes.
    val_idx = list(range(0, len(val_ds), 331))[: args.val_anchors]
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)
    return cfg, applied, ds, policy, pre, post, val_loader, repo_id, val_repo


def train(args) -> dict:
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    torch.manual_seed(args.seed)
    cfg, applied, ds, policy, pre, post, val_loader, repo_id, val_repo = build(args)
    policy.train()

    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr,
                            betas=tuple(cfg.optimizer_betas),
                            weight_decay=cfg.optimizer_weight_decay)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True,
                                         pin_memory=args.device.startswith("cuda"))

    history, running, t0 = [], None, time.time()
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        for key in cfg.image_features:
            if batch[key].dtype == torch.uint8:
                batch[key] = batch[key].float() / 255.0
        batch = pre(batch)
        loss, _ = policy.forward(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        running = loss.item() if running is None else 0.99 * running + 0.01 * loss.item()

        if step % args.val_every == 0 or step == args.steps:
            row = {"step": step, "train_loss": round(running, 5),
                   **validate(policy, pre, post, val_loader, cfg, args.device, args.waypoint)}
            history.append(row)
            print(f"[{args.name}] {row}  ({time.time() - t0:.0f}s)", flush=True)

    del ACTION, OBS_IMAGES
    return {"name": args.name, "overrides": applied, "steps": args.steps,
            "batch_size": args.batch_size, "seed": args.seed, "lr": cfg.optimizer_lr,
            "action_head": cfg.action_head, "use_robot_state": cfg.use_robot_state,
            "action_chunk_size": cfg.action_chunk_size, "n_obs_steps": cfg.n_obs_steps,
            "train_dataset": repo_id, "val_dataset": val_repo, "waypoint": args.waypoint,
            "seconds": round(time.time() - t0, 1), "history": history}


def selftest() -> None:
    from types import SimpleNamespace

    c = SimpleNamespace(use_robot_state=False, action_chunk_size=64)
    assert apply_overrides(c, []) == {}
    assert apply_overrides(c, ["use_robot_state=True"]) == {"use_robot_state": True}
    assert c.use_robot_state is True
    apply_overrides(c, ["action_chunk_size=16"])
    assert c.action_chunk_size == 16 and isinstance(c.action_chunk_size, int)
    for bad in ("no_such_field=1", "novalue"):
        try:
            apply_overrides(c, [bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted")
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="shipped")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--base-config", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--heldout-root", type=Path)
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-anchors", type=int, default=128)
    p.add_argument("--waypoint", type=int, default=8,
                   help="deploy n_action_steps; chunk[waypoint-1] is the only frame the "
                        "Hermite bridge actually takes from the model")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", type=Path)
    p.add_argument("--state-broadcast", action="store_true",
                   help="add the projected state to every patch token, not just one per frame")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.base_config and a.dataset_root and a.heldout_root):
        p.error("--base-config, --dataset-root and --heldout-root are required")
    selftest()
    r = train(a)
    if a.out:
        a.out.write_text(json.dumps(r, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
