#!/usr/bin/env python
"""Turn the 09-02 JSONs into the report's tables.

Three blocks:
  eef        -- both EEF weights on the independent 53-episode eval set (the thing that
                did not exist on 08-31), plus the 08-31 random-split numbers for the same
                patch_policy checkpoint, so the split's difficulty is visible.
  cross      -- joint- and EEF-space policies scored on the SAME physical quantity, by
                pushing joint chunks through the dataset's own forward kinematics.
  reference  -- the joint-space rows from ../scripts_patch_policy_eval_fix, unchanged.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "scripts_patch_policy_eval_fix"
AUG31 = HERE.parent / "scripts_patch_policy_eval_0831"

EEF_GROUPS = {
    "position (m)": [f"eef_{s}_{a}" for s in "lr" for a in ("x", "y", "z")],
    "rotation (rad)": [f"eef_{s}_{a}" for s in "lr" for a in ("roll", "pitch", "yaw")],
    "gripper (0-1)": ["gripper_L", "gripper_R"],
}
ARM = {"arm joints": [f"Joint{i}_{s}" for s in "LR" for i in range(1, 8)],
       "gripper_L": ["gripper_L"], "gripper_R": ["gripper_R"]}


def load(name, d=HERE):
    p = d / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def acc_table(R, cuts=("1", "10", "25", "50"), key="policy_raw"):
    print("\n| run | eval set | " + " | ".join(f"@{c}" for c in cuts)
          + " | rmse | norm_mae | vs null |")
    print("|---|---|" + "---:|" * (len(cuts) + 3))
    for n, (r, tag) in R.items():
        a = r["aggregate"][key]
        null = r["aggregate"]["hold_state"]["mae"]
        print(f"| `{n}` | {tag} | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
              + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} | {null / a['mae']:.2f}x |")
    seen = set()
    for n, (r, tag) in R.items():
        if tag in seen:
            continue
        seen.add(tag)
        a = r["aggregate"]["hold_state"]
        print(f"| *null* `hold_state` | {tag} | " + " | ".join(f"{a['mae_at_horizon'][c]:.5f}" for c in cuts)
              + f" | {a['rmse']:.5f} | {a['norm_mae']:.4f} | - |")


def group_table(R, groups, key="policy_raw"):
    print("\n| run | eval set | " + " | ".join(groups) + " |")
    print("|---|---|" + "---:|" * len(groups))
    rows = list(R.items())
    seen = set()
    for n, (r, tag) in rows:
        if tag not in seen:
            seen.add(tag)
            rows.append((f"*null* hold_state ({tag})",
                         ({"aggregate": {key: r["aggregate"]["hold_state"]}}, tag)))
    for n, (r, tag) in rows:
        pj = r["aggregate"][key]["mae_per_joint"]
        cells = []
        for g, names in groups.items():
            vals = [pj[k] for k in names if k in pj]
            cells.append(f"{sum(vals) / len(vals):.5f}" if vals else "-")
        print(f"| `{n}` | {tag} | " + " | ".join(cells) + " |")


def config_table(R):
    print("| run | type | head | n_obs | state | chunk | train set | anchors | eps | s |")
    print("|---|---|---|---:|---|---:|---|---:|---:|---:|")
    for n, (r, tag) in R.items():
        pd_ = list(r["per_dataset"].values())[0]
        tr = Path(json.load(open(Path(r["checkpoint"]) / "train_config.json"))["dataset"]["repo_id"]).name
        print(f"| `{n}` | {r['policy_type']} | {r.get('policy_action_head') or '-'} | "
              f"{r.get('policy_n_obs_steps')} | {r.get('policy_use_robot_state')} | {r['chunk_size']} | "
              f"`{tr}` | {pd_['anchors']} | {pd_.get('episodes_evaluated', '-')} | {r['total_seconds']:.0f} |")


def noise(R):
    rows = [(n, r) for n, (r, _) in R.items() if "seed_1" in r["aggregate"]]
    if not rows:
        return
    print("\n### sampling noise (same anchors, second diffusion seed)\n")
    print("| run | seed 0 | seed 1 | spread |")
    print("|---|---:|---:|---:|")
    for n, r in rows:
        a, b = r["aggregate"]["policy_raw"]["mae"], r["aggregate"]["seed_1"]["mae"]
        print(f"| `{n}` | {a:.5f} | {b:.5f} | {100 * (b - a) / a:+.1f}% |")


def cross_table(names):
    rows = [(n, load(f"cross_{n}")) for n in names]
    rows = [(n, r) for n, r in rows if r]
    if not rows:
        return
    print("\n# same physical quantity: every chunk pushed through the dataset's own FK\n")
    print("| run | predicts in | position mm | rotation deg | gripper | @1 | @10 | @25 | @50 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for n, r in rows:
        p = r["policy"]
        print(f"| `{n}` | {r['source_space']} | {p['position_mm']:.2f} | {p['rotation_deg']:.3f} | "
              f"{p['gripper']:.5f} | " + " | ".join(f"{p['mae_at_horizon'][h]:.5f}" for h in ("1", "10", "25", "50")) + " |")
    seen = set()
    for n, r in rows:
        if r["source_space"] in seen:
            continue
        seen.add(r["source_space"])
        h = r["hold_state"]
        print(f"| *null* `hold_state` (via the {r['source_space']} eval set) | - | {h['position_mm']:.2f} | {h['rotation_deg']:.3f} | "
              f"{h['gripper']:.5f} | " + " | ".join(f"{h['mae_at_horizon'][x]:.5f}" for x in ("1", "10", "25", "50")) + " |")
    print("\nThe two `hold_state` rows must agree: the joint and EEF eval sets are the same "
          "recordings and FK is deterministic, so 'the arm stays put' is the same trajectory "
          "in both. A gap there means the anchors do not line up.")


def anchor_table():
    rows = [(n, load(f"anchor_{n}")) for n in
            ("pp_eef", "acteef_505", "pp_joint_state5", "act_joint_361")]
    rows = [(n, r) for n, r in rows if r]
    if not rows:
        return
    print("\n# pose anchoring: does the chunk start where the arm is?\n")
    print("| run | space | policy \\|chunk[0]-state\\| | demo \\|action[0]-state\\| | ratio |")
    print("|---|---|---:|---:|---:|")
    for n, r in rows:
        sp = "eef" if r["policy_type"].endswith("eef") or "eef" in r["dataset"] else "joint"
        print(f"| `{n}` | {sp} | {r['policy_mae']:.5f} | {r['demonstration_mae']:.5f} | "
              f"{r['ratio_policy_over_demo']:.3f} |")
    print("\nUnits are the dataset's own (joint radians / metres+radians+gripper), so compare "
          "the ratio column across spaces and the absolute columns only within one.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    NEW = "53-ep independent"
    OLDSET = "67-ep random (08-31)"
    if which in ("all", "eef"):
        R = {n: (r, t) for n, r, t in (
            ("pp_eef_200k", load("pp_eef_200k"), NEW),
            ("pp_eef_100k", load("pp_eef_100k"), NEW),
            ("acteef_505_200k", load("acteef_505_200k"), NEW),
            ("acteef_505_100k", load("acteef_505_100k"), NEW),
            ("acteef_505_h60", load("acteef_505_h60"), NEW),
            ("acteef_361_200k", load("acteef_361_200k"), NEW),
            ("acteef_533_200k", load("acteef_533_200k"), NEW),
            ("pp_eef_200k (08-31)", load("eef_200k", AUG31), OLDSET),
            ("acteef_361 (08-31 fair)", load("acteef_361_fair", AUG31), "18-ep random (08-31)"),
        ) if r}
        print("\n# EEF space, horizon 50 unless noted\n\n## config\n")
        config_table({k: v for k, v in R.items() if "08-31" not in k})
        print("\n## accuracy (MAE over metres + radians + [0,1] gripper -- read the group table)")
        acc_table(R)
        noise(R)
        print("\n## per-group MAE")
        group_table(R, EEF_GROUPS)
    if which in ("all", "anchor"):
        anchor_table()
    if which in ("all", "cross"):
        cross_table(["pp_eef_200k", "acteef_505_200k", "act_joint_361", "pp_joint_state5"])
    if which in ("all", "reference"):
        R = {n: (r, "53-ep independent, joint space") for n, r in (
            ("new_state5", load("new_state5", OLD)), ("act_baseline", load("act_baseline", OLD))) if r}
        print("\n# joint-space reference (unchanged, from ../scripts_patch_policy_eval_fix)")
        acc_table(R)
        group_table(R, ARM)
