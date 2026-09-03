import numpy as np, glob, pandas as pd
SP="/home/kewei/.claude/jobs/b715062d/tmp"
A=np.load(f"{SP}/A200k.npy"); S=np.load(f"{SP}/S200k.npy")
print("=== DEPLOY (200k raw, 52 chunks) ===")
for i,n in [(14,"left_gripper"),(15,"right_gripper")]:
    a=A[:,:,i]; s=S[:,i]
    print(f"{n}: action range [{a.min():+.4f},{a.max():+.4f}]  state range [{s.min():+.4f},{s.max():+.4f}]")
    print(f"    action0 vs state: mean diff={np.mean(A[:,0,i]-s):+.5f}  mean|diff|={np.mean(np.abs(A[:,0,i]-s)):.5f}")
    m=s>1e-6
    if m.sum()>3:
        k=np.polyfit(s[m],A[m,0,i],1)
        print(f"    fit action0 ≈ {k[0]:.3f}*state + {k[1]:+.4f}")
files=sorted(glob.glob("/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_3/data/**/*.parquet",recursive=True))
df=pd.concat([pd.read_parquet(f) for f in files])
Atr=np.stack(df["action"].to_numpy()); Str=np.stack(df["observation.state"].to_numpy())
print("\n=== TRAINING (batch_3) ===")
for i,n in [(14,"left_gripper"),(15,"right_gripper")]:
    print(f"{n}: action range [{Atr[:,i].min():+.4f},{Atr[:,i].max():+.4f}]  state range [{Str[:,i].min():+.4f},{Str[:,i].max():+.4f}]")
    print(f"    action==state? mean|diff|={np.abs(Atr[:,i]-Str[:,i]).mean():.6f}")
