"""Fold the four new-metric runs into the shape of the 09-03 report's tables."""
import json, math, sys
from pathlib import Path

S = Path("/tmp/claude-1000/-home-kewei-YING-paper/ba33f8c8-01a2-43b5-9a62-35695af7732a/scratchpad")
RUNS = [("pp_eef_state", "pp_eef_state_full"), ("pp_eef_nostate", "pp_eef_nostate"),
        ("pp_eef_act5", "pp_eef_act5"), ("acteef_533 (h60)", "acteef_533_h60"),
        ("acteef_533 (h50)", "acteef_533_h50")]
# published policy_raw mae, to prove the ruler did not move
PUB = {"pp_eef_state": 0.03721, "pp_eef_nostate": 0.03916, "pp_eef_act5": 0.03796,
       "acteef_533 (h60)": 0.03689, "acteef_533 (h50)": 0.03443}
POS = [f"eef_{s}_{c}" for s in "lr" for c in "xyz"]
ROT = [f"eef_{s}_{c}" for s in "lr" for c in ("roll", "pitch", "yaw")]

d = {}
for label, stem in RUNS:
    f = S / f"{stem}.json"
    if f.is_file():
        d[label] = json.load(f.open())

print("### reproduction check (policy_raw mae vs the 09-03 report)\n")
print("| run | horizon | this run | report | |")
print("|---|---:|---:|---:|---|")
for label in d:
    a = d[label]["aggregate"]["policy_raw"]["mae"]
    p = PUB[label]
    print(f"| `{label}` | {d[label]['executed_horizon']} | {a:.5f} | {p:.5f} | "
          f"{'MATCH' if abs(a - p) < 5e-5 else f'DIFF {a - p:+.5f}'} |")

print("\n### new: acc@tau, joint/EEF-vector space (fraction of steps inside tau*sigma)\n")
print("| run | @0.1s | @0.25s | @0.5s | @1s | acc@0.25s decay 1 -> 50 |")
print("|---|---:|---:|---:|---:|---|")
for label in d:
    a = d[label]["aggregate"]["policy_raw"]
    t = a["acc_at_tau"]; h = a["acc_at_tau_at_horizon"]
    cuts = sorted(h, key=int)
    print(f"| `{label}` | {t['0.1sigma']:.3f} | {t['0.25sigma']:.3f} | {t['0.5sigma']:.3f} | "
          f"{t['1sigma']:.3f} | {h[cuts[0]]['0.25sigma']:.3f} -> {h[cuts[-1]]['0.25sigma']:.3f} |")

print("\n### new: end-effector pose error (Euclidean / geodesic, both arms averaged)\n")
print("| run | pos (mm) | rot (deg) | report per-axis (mm) | report per-axis (deg) | ratio |")
print("|---|---:|---:|---:|---:|---:|")
for label in d:
    e = d[label]["eef_aggregate"]["policy_raw"]["mean_per_channel"]
    pj = d[label]["aggregate"]["policy_raw"]["mae_per_joint"]
    mm = 1000 * (e["left_pos_m"] + e["right_pos_m"]) / 2
    deg = math.degrees((e["left_rot_rad"] + e["right_rot_rad"]) / 2)
    ax_mm = 1000 * sum(pj[n] for n in POS) / 6
    ax_deg = math.degrees(sum(pj[n] for n in ROT) / 6)
    print(f"| `{label}` | {mm:.2f} | {deg:.2f} | {ax_mm:.2f} | {ax_deg:.2f} | {mm / ax_mm:.2f}x |")

print("\n### new: EEF acc@tau -- fraction of executed steps inside a real tolerance (left arm)\n")
labels = list(d)
taus = list(d[labels[0]]["eef_aggregate"]["policy_raw"]["acc_at_tau"])
print("| run | " + " | ".join(taus) + " |")
print("|---" * (len(taus) + 1) + "|")
for label in labels:
    e = d[label]["eef_aggregate"]["policy_raw"]["acc_at_tau"]
    print(f"| `{label}` | " + " | ".join(f"{e[t]['left_pos_m']:.3f}" for t in taus) + " |")
print("\nnulls, for the same horizon as the run above them:")
for label in labels:
    for null in ("hold_state",):
        e = d[label]["eef_aggregate"][null]["acc_at_tau"]
        print(f"  {label:<18} {null:<11} " + "  ".join(f"{t}={e[t]['left_pos_m']:.3f}" for t in taus))
