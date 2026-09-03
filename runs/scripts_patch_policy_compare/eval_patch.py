"""Open-loop evaluation + vision ablation for two patch_policy checkpoints."""
import json, sys, time
import numpy as np
import torch
from lerobot.policies.patch_policy.modeling_patch_policy import PatchPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

EVAL_ROOT = "/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data"
EVAL_ID = "tidy_up_stationery_le/batch_success_53_eval_data"
J = "/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-"
RUNS = {"A_act": J + "21-58-821581", "B_diffusion": J + "31-19-689756"}
OUT = sys.argv[1]
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 25
MAX_H = 64           # superset horizon pulled from the dataset
FPS = 30
BATCH = 8
IMG_KEYS = ["observation.images.top", "observation.images.wrist_L", "observation.images.wrist_R"]

# ---- conditions: fn(images (B,S,N,C,H,W)) -> images ------------------------------------------
def cond_fns(n_obs):
    def mask_cam(i):
        def f(x):
            x = x.clone(); x[:, :, i] = 0.0; return x
        return f
    return {
        "full":            lambda x: x,
        "zero_images":     lambda x: torch.zeros_like(x),
        "gray_images":     lambda x: torch.full_like(x, 0.5),
        "shuffled_images": lambda x: x.roll(1, dims=0),       # images from another sample
        "mask_top":        mask_cam(0),
        "mask_wrist_L":    mask_cam(1),
        "mask_wrist_R":    mask_cam(2),
        "time_reverse":    lambda x: x.flip(1),
        "freeze_newest":   lambda x: x[:, -1:].expand(-1, n_obs, -1, -1, -1, -1).contiguous(),
    }

def main():
    dt = {k: [i / FPS for i in range(-4, 1)] for k in IMG_KEYS + ["observation.state"]}
    dt["action"] = [i / FPS for i in range(-4, MAX_H)]
    ds = LeRobotDataset(EVAL_ID, root=EVAL_ROOT, delta_timestamps=dt)
    idx = list(range(0, len(ds), STRIDE))
    sub = torch.utils.data.Subset(ds, idx)
    dl = torch.utils.data.DataLoader(sub, batch_size=BATCH, num_workers=8, shuffle=False)
    print(f"eval samples: {len(idx)} / {len(ds)}", flush=True)

    results = {"n_samples": len(idx), "stride": STRIDE, "runs": {}}

    # data-only baselines, gathered once
    gt_all, state_all, ep_all = [], [], []
    for b in dl:
        gt_all.append(b["action"][:, 4:])            # (B, 64, 16) delta 0..63
        state_all.append(b["observation.state"][:, -1])
        ep_all.append(b["episode_index"][:, -1] if b["episode_index"].ndim > 1 else b["episode_index"])
    gt = torch.cat(gt_all); st = torch.cat(state_all); eps = torch.cat(ep_all)
    results["baselines"] = {
        "hold_current_state": err_stats(st.unsqueeze(1).expand_as(gt), gt),
        "dataset_mean_action": err_stats(gt.mean((0, 1))[None, None].expand_as(gt), gt),
    }
    results["gt_action_std_per_step"] = gt.std(0).mean(-1).tolist()

    for tag, ckpt in RUNS.items():
        t0 = time.time()
        pol = PatchPolicy.from_pretrained(ckpt + "/run/checkpoints/200000/pretrained_model").to("cuda").eval()
        cfg = pol.config
        # unnormalizer for actions: reuse the checkpoint's postprocessor stats
        norm = load_action_stats(ckpt + "/run/checkpoints/200000/pretrained_model")
        C = cond_fns(cfg.n_obs_steps)
        preds = {c: [] for c in C}
        torch.manual_seed(0)
        for b in dl:
            imgs = torch.stack([b[k] for k in IMG_KEYS], dim=-4).cuda(non_blocking=True)
            state = b["observation.state"].cuda(non_blocking=True)
            state = normalize(state, norm["state"])
            for cname, fn in C.items():
                with torch.no_grad():
                    torch.manual_seed(0)   # same diffusion noise across conditions
                    out = pol.model.predict({"observation.images": fn(imgs), "observation.state": state})
                preds[cname].append(unnormalize(out, norm["action"]).cpu())
        # diffusion stochasticity check: second seed on `full`
        preds["full_seed1"] = []
        for b in dl:
            imgs = torch.stack([b[k] for k in IMG_KEYS], dim=-4).cuda(non_blocking=True)
            state = normalize(b["observation.state"].cuda(), norm["state"])
            with torch.no_grad():
                torch.manual_seed(1)
                out = pol.model.predict({"observation.images": imgs, "observation.state": state})
            preds["full_seed1"].append(unnormalize(out, norm["action"]).cpu())

        P = {c: torch.cat(v) for c, v in preds.items()}
        H = P["full"].shape[1]
        run = {
            "config": {k: getattr(cfg, k) for k in
                       ["action_head", "action_chunk_size", "n_action_steps", "n_obs_steps",
                        "vision_encoder", "freeze_vision_encoder", "use_robot_state",
                        "optimizer_lr", "dropout"]},
            "horizon": H,
            "params_total": sum(p.numel() for p in pol.parameters()),
            "params_trainable": sum(p.numel() for p in pol.parameters() if p.requires_grad),
            "eval_seconds": None,
            "conditions": {},
        }
        for c, p in P.items():
            g = gt[:, :H]
            s = err_stats(p, g)
            s["pred_shift_vs_full"] = float((p - P["full"]).abs().mean())
            s["mae_per_step"] = p.sub(g).abs().mean((0, 2)).tolist()
            s["mae_per_joint"] = p.sub(g).abs().mean((0, 1)).tolist()
            s["mae_first_k"] = {str(k): float((p[:, :k] - g[:, :k]).abs().mean())
                                for k in [1, 8, 16, 32, 50] if k <= H}
            s["pred_std_over_samples"] = float(p.std(0).mean())
            run["conditions"][c] = s
        # per-episode MAE on the common horizon 50, first n_action_steps
        k = cfg.n_action_steps
        e = (P["full"][:, :k] - gt[:, :k]).abs().mean((1, 2))
        run["per_episode_mae_exec"] = {str(int(i)): float(e[eps == i].mean()) for i in eps.unique()}
        run["eval_seconds"] = round(time.time() - t0, 1)
        results["runs"][tag] = run
        print(f"{tag} done in {run['eval_seconds']}s  full MAE={run['conditions']['full']['mae']:.4f}", flush=True)
        del pol; torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w"), indent=1)
    print("wrote", OUT)

def err_stats(p, g):
    d = (p - g)
    return {"mae": float(d.abs().mean()), "rmse": float(d.pow(2).mean().sqrt()),
            "max_abs": float(d.abs().max()), "mae_step0": float(d[:, 0].abs().mean())}

def load_action_stats(path):
    from safetensors.torch import load_file
    f = load_file(path + "/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    g = load_file(path + "/policy_preprocessor_step_3_normalizer_processor.safetensors")
    def pick(d, key):
        return {k.split(".")[-1]: v for k, v in d.items() if k.startswith(key)}
    return {"action": pick(f, "action"), "state": pick(g, "observation.state")}

def normalize(x, s):
    lo, hi = s["min"].to(x.device), s["max"].to(x.device)
    return (x - lo) / (hi - lo + 1e-8) * 2 - 1

def unnormalize(x, s):
    lo, hi = s["min"].to(x.device), s["max"].to(x.device)
    return (x + 1) / 2 * (hi - lo) + lo

main()
