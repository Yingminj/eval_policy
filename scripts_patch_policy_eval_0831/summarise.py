#!/usr/bin/env python
"""Turn the per-checkpoint report JSONs into the markdown tables of the report.

Two blocks, because the two weights do not share an action space:
  joint  -- 16-D, scored on the 53-episode eval set, comparable to the 08-30 tables
            (those JSONs are read straight out of ../scripts_patch_policy_eval_fix).
  eef    -- 14-D, scored on the 67 held-out episodes of batch_success_533_eef.  MAE
            there averages metres, radians and a [0,1] gripper, so the per-group table
            is the one to read; the scalar is only useful against its own nulls.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "scripts_patch_policy_eval_fix"

EEF_GROUPS = {
    "position (m)": [f"eef_{s}_{a}" for s in "lr" for a in ("x", "y", "z")],
    "rotation (rad)": [f"eef_{s}_{a}" for s in "lr" for a in ("roll", "pitch", "yaw")],
    "gripper (0-1)": ["gripper_L", "gripper_R"],
}


def load(name, d=HERE):
    p = d / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def config_table(R):
    print("| run | head | n_obs | state | chunk | steps | anchors | eps | eval s |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for n, r in R.items():
        pd_ = list(r["per_dataset"].values())[0]
        ck = Path(r["checkpoint"]).parent.name if "checkpoint" in r else "?"
        print(f"| `{n}` | {r.get('policy_action_head') or 'act(policy)'} | {r.get('policy_n_obs_steps')} | "
              f"{r.get('policy_use_robot_state')} | {r['chunk_size']} | {ck} | {pd_['anchors']} | "
              f"{pd_.get('episodes_evaluated', '-')} | {r['total_seconds']:.0f} |")


def acc_table(R, cuts, keys=("policy_raw", "policy_deployed")):
    for key in keys:
        if not any(key in r["aggregate"] for r in R.values()):
            continue
        print(f"\n### {key}\n")
        print("| run | " + " | ".join(f"@{c}" for c in cuts) + " | rmse | norm_mae | vs null |")
        print("|---|" + "---:|" * (len(cuts) + 3))
        for n, r in R.items():
            a = r["aggregate"][key]
            null = r["aggregate"]["hold_state"]["mae"]
            print(f"| `{n}` | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
                  + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} | {null / a['mae']:.2f}x |")
        for null in ("hold_state", "train_mean"):
            a = next(iter(R.values()))["aggregate"][null]
            print(f"| *null* `{null}` | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
                  + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} | - |")


def group_table(R, groups, key="policy_raw"):
    print("\n| run | " + " | ".join(groups) + " |")
    print("|---|" + "---:|" * len(groups))
    rows = dict(R)
    r0 = next(iter(R.values()))
    rows["*null* hold_state"] = {"aggregate": {key: r0["aggregate"]["hold_state"]}}
    for n, r in rows.items():
        pj = r["aggregate"][key]["mae_per_joint"]
        cells = []
        for g, names in groups.items():
            vals = [pj[k] for k in names if k in pj]
            cells.append(f"{sum(vals) / len(vals):.5f}" if vals else "-")
        print(f"| `{n}` | " + " | ".join(cells) + " |")


def ladder(R):
    keys = [k for k in next(iter(R.values()))["aggregate"] if k.startswith("filt_")]
    if not keys:
        return
    print("\n## deploy filter ladder (Δ vs policy_raw)\n")
    print("| stage | " + " | ".join(f"`{n}`" for n in R) + " |")
    print("|---|" + "---:|" * len(R))
    for k in keys:
        row = []
        for n, r in R.items():
            raw = r["aggregate"]["policy_raw"]["mae"]
            v = r["aggregate"][k]["mae"]
            row.append(f"{v:.5f} ({100 * (v - raw) / raw:+.1f}%)")
        print(f"| {k} | " + " | ".join(row) + " |")


def per_horizon(R):
    print("\n## per-horizon MAE, policy_raw\n")
    for n, r in R.items():
        ph = r["aggregate"]["policy_raw"]["mae_per_horizon"]
        idx = [i for i in (0, 4, 9, 19, 29, 39, 49, 59) if i < len(ph)]
        print(f"{n}: " + " ".join(f"{ph[i]:.4f}" for i in idx))
    print("cuts: step 1, 5, 10, 20, 30, 40, 50, 60")


def noise(R):
    rows = [(n, r) for n, r in R.items() if "seed_1" in r["aggregate"]]
    if not rows:
        return
    print("\n## sampling noise (same anchors, second sampler seed)\n")
    print("| run | seed 0 | seed 1 | spread |")
    print("|---|---:|---:|---:|")
    for n, r in rows:
        a, b = r["aggregate"]["policy_raw"]["mae"], r["aggregate"]["seed_1"]["mae"]
        print(f"| `{n}` | {a:.5f} | {b:.5f} | {100 * (b - a) / a:+.1f}% |")


def block(title, R, cuts, groups):
    print(f"\n# {title}\n\n## config\n")
    config_table(R)
    print(f"\n## accuracy (MAE, raw action units)\n")
    acc_table(R, cuts)
    noise(R)
    ladder(R)
    per_horizon(R)
    print("\n## per-group MAE (policy_raw)")
    group_table(R, groups)


ARM = {"arm joints": [f"Joint{i}_{s}" for s in "LR" for i in range(1, 8)],
       "gripper_L": ["gripper_L"], "gripper_R": ["gripper_R"]}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "joint"):
        R = {n: r for n, r in (
            ("vqbet_100k", load("vqbet_100k")), ("vqbet_050k", load("vqbet_050k")),
            ("new_state5", load("new_state5", OLD)), ("new_obs2", load("new_obs2", OLD)),
            ("prev_diffusion", load("prev_diffusion", OLD)),
            ("prev_act_head", load("prev_act_head", OLD)),
            ("act_baseline", load("act_baseline", OLD))) if r}
        block("joint space, horizon 50, 53-episode eval set", R, ["1", "10", "25", "50"], ARM)
        R8 = {n: r for n, r in (
            ("vqbet_100k", load("vqbet_100k_h8")),
            ("new_state5", load("new_state5_h8", OLD)), ("new_obs2", load("new_obs2_h8", OLD)),
            ("prev_diffusion", load("prev_diffusion_h8", OLD)),
            ("prev_act_head", load("prev_act_head_h8", OLD)),
            ("act_baseline", load("act_baseline_h8", OLD))) if r}
        block("joint space, horizon 8 (the deployed window)", R8, ["1"], ARM)
    if which in ("all", "intrain"):
        # Each row against ITS OWN null, because the anchors differ between the two.
        for title, pair, groups in (
            ("joint space, in-train control (batch_success_361, the training set itself)",
             (("vqbet_100k_intrain", load("vqbet_100k_intrain")),), ARM),
            ("EEF space, in-train control (the 468 episodes of 533_eef the weight was trained on)",
             (("eef_200k_intrain", load("eef_200k_intrain")),), EEF_GROUPS),
        ):
            R = {n: r for n, r in pair if r}
            if R:
                block(title, R, ["1", "10", "25", "50"], groups)
    if which in ("all", "eef"):
        R = {n: r for n, r in (("eef_200k", load("eef_200k")), ("eef_100k", load("eef_100k")),
                               ("eef_200k_h60", load("eef_200k_h60"))) if r}
        block("EEF space, 67 held-out episodes of batch_success_533_eef", R,
              ["1", "10", "25", "50"], EEF_GROUPS)
        F = {n: r for n, r in (("eef_200k_fair", load("eef_200k_fair")),
                               ("acteef_361_fair", load("acteef_361_fair")),
                               ("eef_200k_fair_h60", load("eef_200k_fair_h60")),
                               ("acteef_361_fair_h60", load("acteef_361_fair_h60"))) if r}
        if F:
            block("EEF head-to-head, 18 episodes unseen by both checkpoints", F,
                  ["1", "10", "25", "50"], EEF_GROUPS)
