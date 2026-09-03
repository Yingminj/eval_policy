#!/usr/bin/env python
"""Turn the per-checkpoint report JSONs into the markdown tables of the report."""
import json
import sys
from pathlib import Path

ORDER = ["new_state5", "new_obs2", "prev_diffusion", "prev_act_head", "act_baseline"]
d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
sfx = sys.argv[2] if len(sys.argv) > 2 else ""   # "" or "_h8"
R = {n: json.load(open(d / f"{n}{sfx}.json")) for n in ORDER if (d / f"{n}{sfx}.json").exists()}

print("## config\n")
print("| run | head | n_obs | state | chunk | anchors | eval s |")
print("|---|---|---:|---|---:|---:|---:|")
for n, r in R.items():
    pd = list(r["per_dataset"].values())[0]
    print(f"| `{n}` | {r['policy_action_head'] or 'act(policy)'} | {r['policy_n_obs_steps']} | "
          f"{r['policy_use_robot_state']} | {r['chunk_size']} | {pd['anchors']} | {r['total_seconds']:.0f} |")

H = next(iter(R.values()))["executed_horizon"]
print(f"\n## accuracy, executed horizon {H} (raw joint units, MAE)\n")
cuts = [c for c in ("1", "10", "25", "50") if c in next(iter(R.values()))["aggregate"]["policy_raw"]["mae_at_horizon"]]
for key in ("policy_raw", "policy_deployed"):
    print(f"\n### {key}\n")
    print("| run | " + " | ".join(f"@{c}" for c in cuts) + " | rmse | norm_mae |")
    print("|---|" + "---:|" * (len(cuts) + 2))
    for n, r in R.items():
        a = r["aggregate"][key]
        print(f"| `{n}` | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
              + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} |")
    n0 = next(iter(R))
    a = R[n0]["aggregate"]["hold_state"]
    print(f"| *null* `hold_state` | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
          + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} |")

print("\n## sampling noise (same anchors, second diffusion seed)\n")
print("| run | seed 0 | seed 1 | spread |")
print("|---|---:|---:|---:|")
for n, r in R.items():
    if "seed_1" in r["aggregate"]:
        a, b = r["aggregate"]["policy_raw"]["mae"], r["aggregate"]["seed_1"]["mae"]
        print(f"| `{n}` | {a:.5f} | {b:.5f} | {100 * (b - a) / a:+.1f}% |")

print("\n## deploy filter ladder (Δ vs policy_raw)\n")
keys = [k for k in R[next(iter(R))]["aggregate"] if k.startswith("filt_")]
print("| stage | " + " | ".join(f"`{n}`" for n in R) + " |")
print("|---|" + "---:|" * len(R))
for k in keys:
    row = []
    for n, r in R.items():
        raw = r["aggregate"]["policy_raw"]["mae"]
        v = r["aggregate"][k]["mae"]
        row.append(f"{v:.5f} ({100 * (v - raw) / raw:+.1f}%)")
    print(f"| {k} | " + " | ".join(row) + " |")

print("\n## per-horizon MAE, policy_raw (first 50)\n")
for n, r in R.items():
    ph = r["aggregate"]["policy_raw"]["mae_per_horizon"]
    print(f"{n}: " + " ".join(f"{ph[i]:.4f}" for i in [i for i in (0, 4, 9, 19, 29, 39, 49) if i < len(ph)]))
print("cuts: step 1, 5, 10, 20, 30, 40, 50")

print("\n## gripper vs arm (policy_raw, per-joint MAE)\n")
print("| run | arm joints (mean) | gripper_L | gripper_R |")
print("|---|---:|---:|---:|")
for n, r in R.items():
    pj = r["aggregate"]["policy_raw"]["mae_per_joint"]
    arm = [v for k, v in pj.items() if not k.startswith("gripper")]
    print(f"| `{n}` | {sum(arm) / len(arm):.5f} | {pj['gripper_L']:.5f} | {pj['gripper_R']:.5f} |")
