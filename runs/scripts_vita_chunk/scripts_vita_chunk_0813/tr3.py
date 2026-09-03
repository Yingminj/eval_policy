import glob, numpy as np, pandas as pd
files=sorted(glob.glob("/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_3/data/**/*.parquet",recursive=True))
df=pd.concat([pd.read_parquet(f) for f in files])
eps=sorted(df["episode_index"].unique())
print("episodes:",len(eps)," frames:",len(df))
all_d,ac,ratios,profs=[],[],[],[]
for e in eps:
    A=np.stack(df[df["episode_index"]==e]["action"].to_numpy())
    d=np.diff(A,axis=0); all_d.append(np.abs(d))
    for j in range(A.shape[1]):
        x=d[:,j]-d[:,j].mean()
        if np.std(x)>1e-9: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
    for s in range(0,A.shape[0]-8,8):
        w=A[s:s+8]
        path=np.abs(np.diff(w,axis=0)).sum(axis=0); net=np.abs(w[-1]-w[0])
        keep=net>1e-4
        if keep.any(): ratios.append(path[keep]/net[keep])
        sp=np.abs(np.diff(w,axis=0)).sum(axis=1)
        if sp.mean()>1e-9: profs.append(sp/sp.mean())
all_d=np.concatenate(all_d); ratios=np.concatenate(ratios); profs=np.array(profs)
print(f"train |step delta| mean={all_d.mean():.5f} p95={np.percentile(all_d,95):.5f} max={all_d.max():.5f}")
print(f"train lag-1 autocorr mean={np.nanmean(ac):.3f} median={np.nanmedian(ac):.3f}")
print(f"train 8-step path/net moving-only median={np.median(ratios):.2f} p90={np.percentile(ratios,90):.2f}")
print("train normalized speed profile over 8-step windows:")
print("  "+"  ".join(f"{v:.3f}" for v in profs.mean(axis=0)))
