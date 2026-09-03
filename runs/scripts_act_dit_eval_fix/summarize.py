#!/usr/bin/env python3
"""Compact summary per result JSON of offline_chunk_eval.py.

mae_at_horizon = MAE accumulated over the first k executed steps (the headline metric).
mae_per_horizon[k] = MAE of step k alone (shows where in the chunk the error sits).
"""
import json, sys

CUTS = ["1", "10", "25", "50"]
KEYS = ["policy_raw", "policy_deployed", "hold_state"]

for path in sys.argv[1:]:
    d = json.load(open(path))
    a = d["aggregate"]
    n = sum(v.get("anchors", 0) for v in d["per_dataset"].values())
    dropped = sum(v.get("episodes_dropped_as_contaminated", 0) for v in d["per_dataset"].values())
    print(f"== {path.split('/')[-1]}  anchors={n} dropped_ep={dropped} horizon={d['executed_horizon']}")
    print(f"   raw={a['policy_raw']['mae']:.4f}  deployed={a['policy_deployed']['mae']:.4f}  "
          f"null={a['hold_state']['mae']:.4f}  deployed_vs_null={a['hold_state']['mae']/a['policy_deployed']['mae']:.2f}x  "
          f"normMAE(dep)={a['policy_deployed']['norm_mae']:.3f}  no-bridge={a.get('filt_4_excursions',{}).get('mae',float('nan')):.4f}")
    for k in KEYS:
        cum = " ".join(f"{a[k]['mae_at_horizon'][c]:.4f}" for c in CUTS)
        per = " ".join(f"{a[k]['mae_per_horizon'][int(c)-1]:.4f}" for c in CUTS)
        print(f"   {k:<16} cum@{'/'.join(CUTS)}: {cum}   per-step@{'/'.join(CUTS)}: {per}")
