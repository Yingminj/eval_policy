"""Checkpoint sweep, re-anchoring test, motion-scale reference, batch-1 latency."""
import json, os, time, torch
from safetensors.torch import load_file
from lerobot.policies.patch_policy.modeling_patch_policy import PatchPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

J = "/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-"
RUNS = {"A_act": J + "21-58-821581", "B_diffusion": J + "31-19-689756"}
IMG = ["observation.images.top", "observation.images.wrist_L", "observation.images.wrist_R"]
FPS = 30
OUT = "/tmp/claude-1000/-home-kewei-YING-robot-data-platform/38c37621-e06a-4aff-9fb5-e41444b2918b/scratchpad/sweep.json"
ARM, GRIP = slice(0, 14), slice(14, 16)

def stats(path, file, key):
    d = load_file(f"{path}/{file}")
    return {k.split(".")[-1]: v for k, v in d.items() if k.startswith(key)}

dt = {k: [i / FPS for i in range(-4, 1)] for k in IMG + ["observation.state"]}
dt["action"] = [i / FPS for i in range(-4, 64)]
ds = LeRobotDataset("tidy_up_stationery_le/batch_success_53_eval_data",
                    root="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data",
                    delta_timestamps=dt)
sub = torch.utils.data.Subset(ds, list(range(0, len(ds), 100)))
dl = torch.utils.data.DataLoader(sub, batch_size=8, num_workers=8)
print("sweep samples", len(sub), flush=True)

GT, ST = [], []
for b in dl:
    GT.append(b["action"][:, 4:]); ST.append(b["observation.state"][:, -1])
GT = torch.cat(GT); ST = torch.cat(ST)

res = {"n_samples": len(sub), "sweep": {}, "latency": {}, "scale": {}}
hold = ST.unsqueeze(1).expand_as(GT)
res["scale"] = {
    "hold_state_mae_per_step": (hold - GT).abs().mean((0, 2)).tolist(),
    "motion_covered_by_waypoint": {str(k): float((GT[:, k - 1] - ST).abs().mean()) for k in [1, 8, 16, 32, 50, 64]},
    "action_vs_state_step0": float((GT[:, 0] - ST).abs().mean()),
    "arm_motion_32": float((GT[:, 31, ARM] - ST[:, ARM]).abs().mean()),
    "grip_motion_32": float((GT[:, 31, GRIP] - ST[:, GRIP]).abs().mean()),
    "hold_arm_mae_32": float((hold[:, :32, ARM] - GT[:, :32, ARM]).abs().mean()),
    "hold_grip_mae_32": float((hold[:, :32, GRIP] - GT[:, :32, GRIP]).abs().mean()),
    "gripper_switch_frame_fraction": float((GT[:, 1:, GRIP] - GT[:, :-1, GRIP]).abs().gt(0.05).float().mean()),
}
print("scale", json.dumps(res["scale"]["motion_covered_by_waypoint"]), flush=True)

for tag, root in RUNS.items():
    cks = sorted(d for d in os.listdir(root + "/run/checkpoints") if d.isdigit())
    for ck in cks:
        path = f"{root}/run/checkpoints/{ck}/pretrained_model"
        pol = PatchPolicy.from_pretrained(path).to("cuda").eval()
        A = stats(path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors", "action")
        S = stats(path, "policy_preprocessor_step_3_normalizer_processor.safetensors", "observation.state")
        lo, hi = A["min"].cuda(), A["max"].cuda()
        P = []
        for b in dl:
            imgs = torch.stack([b[k] for k in IMG], dim=-4).cuda()
            st = b["observation.state"].cuda()
            st = (st - S["min"].cuda()) / (S["max"].cuda() - S["min"].cuda() + 1e-8) * 2 - 1
            with torch.no_grad():
                torch.manual_seed(0)
                o = pol.model.predict({"observation.images": imgs, "observation.state": st})
            P.append(((o + 1) / 2 * (hi - lo) + lo).cpu())
        P = torch.cat(P); H = P.shape[1]; G = GT[:, :H]
        k = pol.config.n_action_steps
        rean = ST.unsqueeze(1) + (P - P[:, :1])          # shift chunk onto the measured pose
        e = {
            "mae_rad": float((P - G).abs().mean()),
            "mae_step0": float((P[:, 0] - G[:, 0]).abs().mean()),
            "mae_per_step": (P - G).abs().mean((0, 2)).tolist(),
            "mae_exec_window": float((P[:, :k] - G[:, :k]).abs().mean()),
            "reanchored_mae": float((rean - G).abs().mean()),
            "reanchored_exec": float((rean[:, :k] - G[:, :k]).abs().mean()),
            "pose_bias_vs_state": float((P[:, 0] - ST).abs().mean()),
            "arm_mae_exec": float((P[:, :k, ARM] - G[:, :k, ARM]).abs().mean()),
            "grip_mae_exec": float((P[:, :k, GRIP] - G[:, :k, GRIP]).abs().mean()),
            "vs_hold": float((hold[:, :H] - G).abs().mean() / (P - G).abs().mean()),
        }
        res["sweep"].setdefault(tag, {})[ck] = e
        print(tag, ck, {q: round(v, 4) for q, v in e.items() if isinstance(v, float)}, flush=True)
        if ck == cks[-1]:
            bb = {"observation.images": torch.rand(1, 5, 3, 3, 480, 640, device="cuda"),
                  "observation.state": torch.zeros(1, 5, 16, device="cuda")}
            with torch.no_grad():
                pol.model.predict(bb); torch.cuda.synchronize()
                t = time.time()
                for _ in range(5): pol.model.predict(bb)
                torch.cuda.synchronize()
            lat = (time.time() - t) / 5
            res["latency"][tag] = {"batch1_s": round(lat, 4), "n_action_steps": k,
                                   "exec_window_s": round(k / FPS, 3), "duty_cycle": round(lat / (k / FPS), 3)}
            print("latency", tag, res["latency"][tag], flush=True)
        del pol; torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=1); print("wrote", OUT)
