#!/usr/bin/env python3
"""Turn results/*.json into the tables the report uses.  Numbers only, no interpretation.

    /opt/robot-platform/train-venv/bin/python summarise.py           > tables.md
    /opt/robot-platform/train-venv/bin/python summarise.py --selftest

Every number in the report must come out of here.  Nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"

# 200k arms, in the order the report reads them.
ORDER = ["pp_eef_state", "pp_eef_nostate", "pp_eef_act5", "acteef_533"]
LABEL = {
    "pp_eef_state": "pp_eef_state (08-31)",
    "pp_eef_nostate": "pp_eef_nostate (09-44)",
    "pp_eef_act5": "pp_eef_act5 (09-33)",
    "acteef_533": "acteef_533 (deployed)",
}
# Published numbers this run has to reproduce before any of its own are worth reading.
# Both from patch_policy-eef-independent-eval-2026-09.md §4, horizon 50, policy_raw.
PUBLISHED = {"pp_eef_state_h60": 0.03721, "acteef_533_h50": 0.03443}

POS, ROT = ["eef_l_x", "eef_l_y", "eef_l_z", "eef_r_x", "eef_r_y", "eef_r_z"], \
           ["eef_l_roll", "eef_l_pitch", "eef_l_yaw", "eef_r_roll", "eef_r_pitch", "eef_r_yaw"]
GRIP = ["gripper_L", "gripper_R"]


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def cell(v, w=8, p=5):
    return f"{v:{w}.{p}f}" if v is not None else f"{'—':>{w}}"


def group(agg, key="mae_per_joint"):
    """Raw units are metres and radians in one vector; report them apart, in mm and degrees."""
    import math
    d = agg[key]
    mm = 1000 * sum(d[n] for n in POS) / len(POS)
    deg = math.degrees(sum(d[n] for n in ROT) / len(ROT))
    grip = sum(d[n] for n in GRIP) / len(GRIP)
    return mm, deg, grip


def main(argv):
    runs = {n: load(f"{n}_h60") for n in ORDER}
    runs = {n: d for n, d in runs.items() if d}
    if not runs:
        sys.exit("no results/*_h60.json yet -- run ./run_eval.sh first")

    first = next(iter(runs.values()))
    ds = first["per_dataset"]["tidy_up_stationery_le/batch_success_53_eval_data_eef"]
    print(f"eval set: {ds['episodes_evaluated']} episodes / {ds['frames']} frames / "
          f"{ds['anchors']} anchors / {ds['anchor_action_steps']} scored action steps, "
          f"stride {first['stride']}, contaminated episodes dropped: "
          f"{ds['episodes_dropped_as_contaminated']}")
    print(f"deploy filters: {first['deploy_filters']}   batch_size {first['batch_size']}   "
          f"seed {first['seed']}")

    print("\n### 1. main table — executed window (`--n-action-steps 60`, clamped to each policy's chunk)\n")
    print("| run | head | n_obs | state | horizon | raw | deployed | @1 | @10 | @25 | @50 | "
          "raw vs null | deployed vs null | 墙钟 (s) | 每 anchor (ms) |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, d in runs.items():
        a, hz = d["aggregate"], d["aggregate"]["policy_raw"]["mae_at_horizon"]
        null = a["hold_state"]["mae"]
        na = d["per_dataset"]["tidy_up_stationery_le/batch_success_53_eval_data_eef"]["anchors"]
        # NOTE: wall clock includes data loading and, where seed_repeat=1, TWO forward passes
        # per anchor.  It is a relative cost, not an isolated inference latency.
        print(f"| `{LABEL[n]}` | {d['policy_action_head'] or 'act(ACT)'} | {d['policy_n_obs_steps']} | "
              f"{'on' if d['policy_use_robot_state'] else 'off'} | {d['executed_horizon']} | "
              f"{cell(a['policy_raw']['mae'])} | {cell(a['policy_deployed']['mae'])} | "
              f"{cell(hz['1'])} | {cell(hz['10'])} | {cell(hz['25'])} | {cell(hz.get('50'))} | "
              f"{null / a['policy_raw']['mae']:.2f}x | {null / a['policy_deployed']['mae']:.2f}x | "
              f"{d['total_seconds']:.0f}{' (2 draws)' if d['seed_repeat'] else ''} | "
              f"{1000 * d['total_seconds'] / na / (1 + d['seed_repeat']):.1f} |")
    print("\n| null baseline | mae | @1 | @10 | @25 | @50 |")
    print("|---|---:|---:|---:|---:|---:|")
    seen = set()  # one null row per distinct horizon; runs sharing a horizon share the null
    for n, d in runs.items():
        h = d["executed_horizon"]
        if h in seen:
            continue
        seen.add(h)
        for k in ("hold_state", "train_mean"):
            a = d["aggregate"][k]
            hz = a["mae_at_horizon"]
            print(f"| `{k}` @ horizon {h} | {cell(a['mae'])} | "
                  f"{cell(hz['1'])} | {cell(hz['10'])} | {cell(hz['25'])} | {cell(hz.get('50'))} |")

    print("\n**相对基线 `pp_eef_state` 的差值**（同一 anchor 集、同一 seed、同一 batch size）：\n")
    print("| run | raw Δ | deployed Δ | @1 Δ | 位置 Δ | 采样噪声下限 |")
    print("|---|---:|---:|---:|---:|---:|")
    base = runs["pp_eef_state"]["aggregate"]
    bmm = group(base["policy_raw"])[0]
    for n, d in runs.items():
        if n == "pp_eef_state":
            continue
        a = d["aggregate"]
        floor = (f"{abs(a['seed_1']['mae'] / a['policy_raw']['mae'] - 1) * 100:.2f}%"
                 if "seed_1" in a and a["seed_1"]["mae"] != a["policy_raw"]["mae"] else "0（确定性）")
        # a row scored over a different horizon is not comparable to the baseline row
        warn = "" if d["executed_horizon"] == runs["pp_eef_state"]["executed_horizon"] \
            else f" **(h{d['executed_horizon']}，跨 horizon，不可直接读，见 §7)**"
        print(f"| `{LABEL[n]}`{warn} | {(a['policy_raw']['mae'] / base['policy_raw']['mae'] - 1) * 100:+.1f}% | "
              f"{(a['policy_deployed']['mae'] / base['policy_deployed']['mae'] - 1) * 100:+.1f}% | "
              f"{(a['policy_raw']['mae_at_horizon']['1'] / base['policy_raw']['mae_at_horizon']['1'] - 1) * 100:+.1f}% | "
              f"{(group(a['policy_raw'])[0] / bmm - 1) * 100:+.1f}% | {floor} |")
    a5, an = runs["pp_eef_act5"]["aggregate"], runs["pp_eef_nostate"]["aggregate"]
    print(f"\n`pp_eef_act5` vs `pp_eef_nostate`（两者 `use_robot_state` 都是 off，差的是 head 与 n_obs）："
          f"raw {(a5['policy_raw']['mae'] / an['policy_raw']['mae'] - 1) * 100:+.1f}%, "
          f"deployed {(a5['policy_deployed']['mae'] / an['policy_deployed']['mae'] - 1) * 100:+.1f}%, "
          f"墙钟 {runs['pp_eef_act5']['total_seconds'] / runs['pp_eef_nostate']['total_seconds']:.2f}x")

    print("\n### 2. rmse and tail\n")
    print("| run | raw rmse | deployed rmse | norm_mae | norm_rmse | tail ratio |")
    print("|---|---:|---:|---:|---:|---:|")
    for n, d in runs.items():
        r, dep = d["aggregate"]["policy_raw"], d["aggregate"]["policy_deployed"]
        print(f"| `{LABEL[n]}` | {cell(r['rmse'])} | {cell(dep['rmse'])} | {cell(r['norm_mae'])} | "
              f"{cell(r['norm_rmse'])} | {r['tail_ratio']:.2f} |")

    print("\n### 3. grouped error — the 14 dims do not share a unit\n")
    print("| run | 位置 MAE (mm) | 姿态 MAE (°) | 夹爪 MAE (0-1) | 夹爪 clip 后 | 位置 deployed (mm) | 姿态 deployed (°) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for n, d in list(runs.items()) + [("hold_state", first), ("train_mean", first)]:
        key = n if n in ("hold_state", "train_mean") else "policy_raw"
        mm, deg, g = group(d["aggregate"][key])
        if key == "policy_raw":
            mm2, deg2, _ = group(d["aggregate"]["policy_deployed"])
            gc = group(d["aggregate"]["filt_0_clip_only"])[2]
            print(f"| `{LABEL[n]}` | {mm:.2f} | {deg:.2f} | {g:.4f} | {gc:.4f}"
                  f"{'（未越界）' if gc == g else ''} | {mm2:.2f} | {deg2:.2f} |")
        else:
            print(f"| *null* `{n}` (h{first['executed_horizon']}) | {mm:.2f} | {deg:.2f} | {g:.4f} | — | — | — |")

    print("\n### 4. sampling noise — is any of the above bigger than the head's own variance?\n")
    print("| run | seed 0 (raw) | seed 1 | Δ | 确定性? |")
    print("|---|---:|---:|---:|---|")
    for n, d in runs.items():
        a = d["aggregate"]
        if "seed_1" not in a:
            print(f"| `{LABEL[n]}` | {cell(a['policy_raw']['mae'])} | — | — | 未测（ACT，无采样） |")
            continue
        s0, s1 = a["policy_raw"]["mae"], a["seed_1"]["mae"]
        det = "**是**（逐位相同）" if s0 == s1 else f"否（{abs(s1 / s0 - 1) * 100:.2f}%）"
        print(f"| `{LABEL[n]}` | {cell(s0)} | {cell(s1)} | {(s1 - s0) * 1e3:+.3f}e-3 | {det} |")

    print("\n### 5. deploy filter ladder (cumulative, deploy order)\n")
    print("`gripper_loops` and `excursions` are ENABLE_*=False on the robot, so their rungs repeat "
          "the previous one by construction.\n")
    stages = ["policy_raw", "filt_0_clip_only", "filt_1_rollbacks", "filt_2_gripper_loops",
              "filt_3_smoothing", "filt_4_excursions", "filt_5_bridge", "filt_bridge_only"]
    print("| run | " + " | ".join(s.replace("filt_", "").replace("policy_", "") for s in stages) + " |")
    print("|---|" + "---:|" * len(stages))
    for n, d in runs.items():
        a = d["aggregate"]
        print(f"| `{LABEL[n]}` | " + " | ".join(f"{a[s]['mae']:.5f}" if s in a else "—" for s in stages) + " |")
    print("\n相对 `policy_raw` 的百分比：\n")
    print("| run | " + " | ".join(s.replace("filt_", "") for s in stages[1:]) + " |")
    print("|---|" + "---:|" * (len(stages) - 1))
    for n, d in runs.items():
        a = d["aggregate"]
        base = a["policy_raw"]["mae"]
        print(f"| `{LABEL[n]}` | " + " | ".join(
            f"{(a[s]['mae'] / base - 1) * 100:+.1f}%" if s in a else "—" for s in stages[1:]) + " |")

    print("\n### 6. 100k vs 200k — is either new arm still improving?\n")
    print("| run | 100k raw | 200k raw | Δ | 100k deployed | 200k deployed |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in ("pp_eef_nostate", "pp_eef_act5"):
        a, b = load(f"{n}_h60_100k"), load(f"{n}_h60")
        if not (a and b):
            continue
        x, y = a["aggregate"]["policy_raw"]["mae"], b["aggregate"]["policy_raw"]["mae"]
        print(f"| `{LABEL[n]}` | {cell(x)} | {cell(y)} | {(y / x - 1) * 100:+.1f}% | "
              f"{cell(a['aggregate']['policy_deployed']['mae'])} | "
              f"{cell(b['aggregate']['policy_deployed']['mae'])} |")

    print("\n### 7. horizon 50 vs 60 for the ACT baseline (patch_policy cannot reach 60)\n")
    print("| run | horizon | chunk | raw | deployed | hold_state | raw vs null | deployed vs null | 位置 (mm) | 姿态 (°) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in ("acteef_533_h50", "acteef_533_h60"):
        d = load(k)
        if not d:
            continue
        a = d["aggregate"]
        mm, deg, g = group(a["policy_raw"])
        print(f"| `{k}` | {d['executed_horizon']} | {d['chunk_size']} | {cell(a['policy_raw']['mae'])} | "
              f"{cell(a['policy_deployed']['mae'])} | {cell(a['hold_state']['mae'])} | "
              f"{a['hold_state']['mae'] / a['policy_raw']['mae']:.2f}x | "
              f"{a['hold_state']['mae'] / a['policy_deployed']['mae']:.2f}x | {mm:.2f} | {deg:.2f} |")

    if "--selftest" in argv:
        for name, want in PUBLISHED.items():
            d = load(name)
            assert d, f"{name}.json missing"
            got = d["aggregate"]["policy_raw"]["mae"]
            assert abs(got - want) < 5e-5, f"{name}: {got:.5f} != published {want} -- not the same ruler"
        d = load("pp_eef_act5_h60")
        if d:
            a = d["aggregate"]
            assert a["policy_raw"]["mae"] == a["seed_1"]["mae"], \
                "the act head produced two different draws; it is not deterministic"
        print("\nselftest OK: published pp_eef/acteef_533 numbers reproduce; act head is deterministic")


if __name__ == "__main__":
    main(sys.argv[1:])
