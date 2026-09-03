import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob("/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_2/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files])
eps = sorted(df["episode_index"].unique())
print("episodes:", len(eps), " frames:", len(df))

all_d, autocorrs, ratios = [], [], []
for e in eps:
    A = np.stack(df[df["episode_index"] == e]["action"].to_numpy())
    d = np.diff(A, axis=0)
    all_d.append(np.abs(d))
    for j in range(A.shape[1]):
        x = d[:, j] - d[:, j].mean()
        if np.std(x) > 1e-9:
            autocorrs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    for s in range(0, A.shape[0] - 8, 8):
        w = A[s:s+8]
        path = np.abs(np.diff(w, axis=0)).sum(axis=0)
        net = np.abs(w[-1] - w[0])
        keep = net > 1e-4          # ignore windows where the joint is stationary
        ratios.append((path[keep] / net[keep]))

all_d = np.concatenate(all_d)
ratios = np.concatenate(ratios)
print(f"train |step delta|  mean={all_d.mean():.5f}  p95={np.percentile(all_d,95):.5f}  max={all_d.max():.5f}")
print(f"train lag-1 autocorr: mean={np.nanmean(autocorrs):.3f}  median={np.nanmedian(autocorrs):.3f}")
print(f"train 8-step path/net (moving joints only): median={np.median(ratios):.2f}  p90={np.percentile(ratios,90):.2f}")
