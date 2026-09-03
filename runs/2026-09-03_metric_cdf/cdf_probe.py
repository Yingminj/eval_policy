#!/usr/bin/env python3
"""从 --trace 导出的 npz 直接读误差 CDF，用来决定要不要给 harness 加阈值指标。

用法: python cdf_probe.py <trace.npz> <对应的 aggregate.json>
每关节 sigma 从报告反解: sigma_j = mae_per_joint[j] / norm_mae_per_joint[j]
acc@tau 只在臂关节上报（D3：夹爪 R≈4.2，会把表头虚高约 6 pp）。
"""
import json
import sys

import numpy as np

TAUS = (0.05, 0.1, 0.2, 0.5)


def main(trace, report):
    a = json.load(open(report))["aggregate"]
    a = a.get("policy") or a["policy_raw"]
    names = list(a["mae_per_joint"])
    sig = np.array([a["mae_per_joint"][n] / max(a["norm_mae_per_joint"][n], 1e-12) for n in names])

    d = np.load(trace)
    assert list(d["joint_names"]) == names, "trace 与报告的关节顺序不一致"
    gt, val = d["gt"], d["valid"]
    arm = [i for i, n in enumerate(names) if not n.startswith("gripper")]  # D3

    def row(label, pred):
        e = (np.abs(pred - gt) / sig)[val]  # D6: 只取 valid 槽，padding 的 0 误差会虚高 acc@tau
        ea = e[:, arm].ravel()
        e = e.ravel()
        print(
            f"{label:<12} norm_mae={e.mean():.4f} R={np.sqrt((e**2).mean())/e.mean():.3f} "
            f"p50={np.median(e):.4f} p90={np.percentile(e, 90):.4f} | "
            + " ".join(f"@{t}={100*(ea < t).mean():.1f}%" for t in TAUS)
            + "  (臂关节)"
        )

    row("policy", d["pred"])
    row("hold_state", np.repeat(d["state"][:, None, :], gt.shape[1], axis=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
