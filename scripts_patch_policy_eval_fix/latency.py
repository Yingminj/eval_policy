#!/usr/bin/env python
"""Batch-1 inference latency, against the window the chunk has to cover.

A chunk of `n_action_steps` waypoints played at 30 Hz buys `n_action_steps / 30` seconds
before the next one is needed (`_check_need_new_chunk` fires only once the whole chunk has
been played). `duty` is the fraction of that window one inference eats; above 1.0 the policy
cannot keep the arm fed and the robot stalls between chunks.

Run with the GPU otherwise idle -- a second process inflates every number here.

    /opt/robot-platform/train-venv/bin/python latency.py --out latency.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

J = Path("/mnt/robot_platform/jobs")
RUNS = {
    "new_state5": J / "patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413",
    "new_obs2": J / "patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-31-30-522146",
    "prev_diffusion": J / "patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756",
    "prev_act_head": J / "patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581",
    "act_baseline": J / "act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328",
}


def bench(name: str, ckpt: Path, reps: int = 20) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from offline_chunk_eval import load_policy

    policy, _, _, cfg = load_policy(ckpt, "cuda")
    cams = list(cfg.image_features)
    h, w = next(iter(cfg.image_features.values())).shape[-2:]

    if cfg.type == "patch_policy":
        s = cfg.n_obs_steps
        batch = {
            "observation.images": torch.rand(1, s, len(cams), 3, h, w, device="cuda"),
            "observation.state": torch.zeros(1, s, cfg.robot_state_feature.shape[0], device="cuda"),
        }
        call = lambda: policy.model.predict(batch)  # noqa: E731
    else:
        batch = {c: torch.rand(1, 3, h, w, device="cuda") for c in cams}
        batch["observation.state"] = torch.zeros(1, cfg.robot_state_feature.shape[0], device="cuda")
        call = lambda: policy.predict_action_chunk(batch)  # noqa: E731

    with torch.no_grad():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            call()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    ts.sort()
    k = cfg.n_action_steps
    med = ts[len(ts) // 2]
    out = {
        "median_s": round(med, 4),
        "p10_s": round(ts[int(0.1 * len(ts))], 4),
        "p90_s": round(ts[int(0.9 * len(ts))], 4),
        "n_action_steps": k,
        "n_obs_steps": getattr(cfg, "n_obs_steps", 1),
        "encoder_images_per_call": getattr(cfg, "n_obs_steps", 1) * len(cams),
        "window_s": round(k / 30, 3),
        "duty": round(med / (k / 30), 3),
        "denoise_steps": getattr(getattr(policy, "model", None), "num_inference_steps", None),
    }
    del policy
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("latency.json"))
    p.add_argument("--only", action="append", default=[])
    args = p.parse_args()

    res = {}
    for name, job in RUNS.items():
        if args.only and name not in args.only:
            continue
        res[name] = bench(name, job / "run/checkpoints/200000/pretrained_model")
        print(name, res[name], flush=True)
    args.out.write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
