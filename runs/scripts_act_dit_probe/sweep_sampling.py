#!/usr/bin/env python
"""Two inference-side knobs for a trained ACT-DiT checkpoint, scored offline. No retraining.

`act_dit` emits a chunk by integrating a flow-matching ODE from a fresh Gaussian draw.  Two
things follow that plain ACT does not have, and both are visible in the offline numbers as
an error floor that does not shrink with the horizon:

  * the sample is only as accurate as the integrator.  `--steps` sweeps
    `num_integration_steps` (flow matching, 10 Euler steps by default) or
    `num_inference_steps` (diffusion, 10 DDIM steps by default), whichever objective the
    checkpoint was trained with.
  * the sample is *stochastic*.  A different noise draw gives a different chunk, and at the
    first action of the chunk -- the one the robot executes immediately -- that variance is
    pure error, because the correct first action is nearly determined by the current pose.

This script sweeps both against the same held-out anchors, reusing `offline_chunk_eval`'s
accumulator, contamination filter and `hold_state` null baseline so the numbers line up with
the reports already written against that script.  `mae@1` and `mae@10` are the columns to
read: they are what the robot executes before the next chunk arrives.

Usage
-----
    PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src:../scripts_act_eval_test \
    python sweep_sampling.py \
        --checkpoint /mnt/.../checkpoints/200000/pretrained_model \
        --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_1 \
        --train-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361 \
        --steps 10 --steps 50 --samples 1 --samples 8 --out sweep.json

Self-check:  python sweep_sampling.py --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import offline_chunk_eval as oce


def averaged_chunk(policy, n_samples: int):
    """Wrap `predict_action_chunk` so it returns the mean of `n_samples` independent draws.

    The mean of several ODE samples is not itself a sample of the learned distribution -- on
    a genuinely multimodal action distribution it interpolates between modes, which is the
    classic reason not to average a generative policy.  It is measured here anyway because
    the failure being chased is the opposite one: a near-deterministic target buried under
    sampling variance.  If averaging helps, the variance was noise, not modes.
    """
    inner = policy.predict_action_chunk

    def wrapped(batch):
        return torch.stack([inner(batch) for _ in range(n_samples)]).mean(dim=0)

    return wrapped


def sweep(checkpoint: Path, dataset_roots: list[Path], train_root: Path | None,
          steps_list: list[int], samples_list: list[int], stride: int, batch_size: int,
          num_workers: int, device: str, max_anchors: int | None) -> dict:
    base_load = oce.load_policy
    results = {}

    for n_steps in steps_list:
        for n_samples in samples_list:
            key = f"steps={n_steps},samples={n_samples}"

            def patched(ck, dev, _s=n_steps, _n=n_samples):
                policy, pre, post, cfg = base_load(ck, dev)
                if cfg.objective == "flow_matching":
                    cfg.num_integration_steps = _s
                    policy.config.num_integration_steps = _s
                    # The objective reads the value off the config it was handed at
                    # construction, which is the same object -- but assert it rather
                    # than assume it.
                    assert policy.objective.config.num_integration_steps == _s
                else:
                    # `DiffusionObjective` resolves `num_inference_steps` once in its
                    # __init__ and caches it on itself, so the config alone is not enough.
                    cfg.num_inference_steps = _s
                    policy.config.num_inference_steps = _s
                    policy.objective.num_inference_steps = _s
                    assert policy.objective.num_inference_steps == _s
                if _n > 1:
                    policy.predict_action_chunk = averaged_chunk(policy, _n)
                return policy, pre, post, cfg

            oce.load_policy = patched
            try:
                r = oce.evaluate(
                    checkpoint=checkpoint, dataset_roots=dataset_roots, stride=stride,
                    batch_size=batch_size, num_workers=num_workers, device=device,
                    max_anchors_per_dataset=max_anchors, train_root=train_root,
                )
            finally:
                oce.load_policy = base_load

            agg = r["aggregate"]
            results[key] = {
                "policy": {k: round(v, 6) for k, v in agg["policy"]["mae_at_horizon"].items()},
                "policy_mae": round(agg["policy"]["mae"], 6),
                "hold_state": {k: round(v, 6) for k, v in agg["hold_state"]["mae_at_horizon"].items()},
                "anchor_action_steps": agg["policy"]["anchor_action_steps"],
            }
            print(f"[{key}] mae@1={results[key]['policy']['1']:.5f} "
                  f"mae@10={results[key]['policy']['10']:.5f} "
                  f"mae={results[key]['policy_mae']:.5f}", flush=True)
    return {"checkpoint": str(checkpoint), "stride": stride, "results": results}


def selftest() -> None:
    """`averaged_chunk` must average, and must leave a 1-sample path untouched."""
    from types import SimpleNamespace

    calls = {"n": 0}

    def fake(_batch):
        calls["n"] += 1
        return torch.full((1, 2, 3), float(calls["n"]))

    p = SimpleNamespace(predict_action_chunk=fake)
    out = averaged_chunk(p, 4)(None)
    assert calls["n"] == 4, calls
    assert torch.allclose(out, torch.full((1, 2, 3), 2.5)), out  # mean of 1,2,3,4
    assert out.shape == (1, 2, 3), out.shape
    print("selftest OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset-root", type=Path, action="append", default=[])
    p.add_argument("--train-root", type=Path, default=None)
    p.add_argument("--steps", type=int, action="append", default=[])
    p.add_argument("--samples", type=int, action="append", default=[])
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-anchors-per-dataset", type=int, default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.checkpoint or not a.dataset_root:
        p.error("--checkpoint and at least one --dataset-root are required")
    selftest()
    r = sweep(a.checkpoint, a.dataset_root, a.train_root, a.steps or [10],
              a.samples or [1], a.stride, a.batch_size, a.num_workers, a.device,
              a.max_anchors_per_dataset)
    print(json.dumps(r["results"], indent=2))
    if a.out:
        a.out.write_text(json.dumps(r, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
