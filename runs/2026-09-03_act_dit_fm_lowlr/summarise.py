#!/usr/bin/env python3
"""Table the results/*.json produced by run.sh.  Numbers only, no interpretation."""
import json, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ORDER = ["act_baseline", "fm_lowlr", "fm_lowlr_noadaln_rk4", "fm_hilr", "diff_lowlr"]


def load(name, h):
    p = R / f"eval53_{name}_h{h}.json"
    return json.load(open(p)) if p.exists() else None


def row(d, pred):
    a = d["aggregate"][pred]
    hz = a["mae_at_horizon"]
    return a["mae"], a["rmse"], a["norm_mae"], hz


print(f"\n### horizon 100 — policy_raw (ruler of act_dit-lowlr §4)")
print(f"{'run':24s} {'mae':>8s} {'rmse':>8s} {'norm':>7s} {'@1':>8s} {'@10':>8s} {'@25':>8s} {'@50':>8s} {'vs null':>8s}")
for n in ORDER + ["hold_state"]:
    d = load(ORDER[0] if n == "hold_state" else n, 100)
    if not d:
        continue
    pred = "hold_state" if n == "hold_state" else "policy_raw"
    mae, rmse, nm, hz = row(d, pred)
    null = d["aggregate"]["hold_state"]["mae"]
    print(f"{n:24s} {mae:8.5f} {rmse:8.5f} {nm:7.4f} "
          f"{hz['1']:8.5f} {hz['10']:8.5f} {hz['25']:8.5f} {hz['50']:8.5f} {null/mae:7.2f}x")

print(f"\n### horizon 50 — policy_raw / policy_deployed (ruler of act_dit-flowmatching-deployed §3)")
print(f"{'run':24s} {'raw':>8s} {'deployed':>9s} {'@1':>8s} {'@10':>8s} {'bridge Δ':>9s} {'vs null':>8s}")
for n in ORDER + ["hold_state"]:
    d = load(ORDER[0] if n == "hold_state" else n, 50)
    if not d:
        continue
    a = d["aggregate"]
    null = a["hold_state"]["mae"]
    if n == "hold_state":
        hz = a["hold_state"]["mae_at_horizon"]
        print(f"{n:24s} {a['hold_state']['mae']:8.5f} {'—':>9s} {hz['1']:8.5f} {hz['10']:8.5f} {'—':>9s} {'—':>8s}")
        continue
    raw, dep = a["policy_raw"]["mae"], a["policy_deployed"]["mae"]
    hz = a["policy_raw"]["mae_at_horizon"]
    print(f"{n:24s} {raw:8.5f} {dep:9.5f} {hz['1']:8.5f} {hz['10']:8.5f} "
          f"{(dep/raw-1)*100:+8.1f}% {null/dep:7.2f}x")

# acceptance line from act_dit-lowlr §6.5: eval53 mae@10 must beat hold_state's
print("\n### acceptance line: mae@10 < hold_state mae@10 (horizon-50 anchors)")
d0 = load(ORDER[0], 50)
line = d0["aggregate"]["hold_state"]["mae_at_horizon"]["10"]
for n in ORDER:
    d = load(n, 50)
    if not d:
        continue
    v = d["aggregate"]["policy_raw"]["mae_at_horizon"]["10"]
    print(f"{n:24s} mae@10={v:.5f}  {'PASS' if v < line else 'FAIL'}  (line {line:.5f})")

if __name__ == "__main__" and "--selftest" in sys.argv:
    d = load("act_baseline", 100)
    assert abs(d["aggregate"]["policy_raw"]["mae"] - 0.0685) < 5e-4, "ACT baseline drifted from the published 0.0685"
    print("\nselftest OK: ACT baseline reproduces the published h100 number")
