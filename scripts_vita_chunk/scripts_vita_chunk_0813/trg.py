import glob, numpy as np, pandas as pd
files=sorted(glob.glob("/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_3/data/**/*.parquet",recursive=True))
df=pd.concat([pd.read_parquet(f) for f in files])
rev=np.zeros(16); tot=np.zeros(16); ac=[[] for _ in range(16)]
for e in sorted(df["episode_index"].unique()):
    A=np.stack(df[df["episode_index"]==e]["action"].to_numpy())
    d=np.diff(A,axis=0)
    f=(np.sign(d[1:])!=np.sign(d[:-1]))&(np.abs(d[1:])>1e-4)
    rev+=f.sum(axis=0); tot+=f.shape[0]
    for j in range(16):
        x=d[:,j]-d[:,j].mean()
        if np.std(x)>1e-9: ac[j].append(np.corrcoef(x[:-1],x[1:])[0,1])
J=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
print("TRAINING batch_3 reference, per joint:")
for j in range(16): print(f"  {J[j]:<22} rev={rev[j]/tot[j]*100:>5.1f}%  ac={np.nanmean(ac[j]):+.3f}")
print(f"\narms   rev={rev[:14].sum()/tot[:14].sum()*100:.1f}%  ac={np.nanmean([v for s in ac[:14] for v in s]):+.3f}")
print(f"grippers rev={rev[14:].sum()/tot[14:].sum()*100:.1f}%  ac={np.nanmean([v for s in ac[14:] for v in s]):+.3f}")
