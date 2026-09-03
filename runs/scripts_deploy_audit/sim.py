import sys, math
import numpy as np, torch
from traj_standalone import (cubic_hermite_segment, remove_boundary_rollbacks,
    remove_small_rollbacks, remove_open_gripper_loops, smooth_action_chunk, smooth_large_excursions)

SP="/home/kewei/YING/paper/policy/scripts_vita_chunk/scripts_vita_chunk_0813pm/"
A=np.load(SP+"A200k.npy"); S=np.load(SP+"S200k.npy")
print("A",A.shape,"S",S.shape)

thr=0.0087
d0 = np.abs(A[:,0,:14]-S[:,:14])
maxdiff = d0.max(axis=1)
print("max|action0-state| over 14 arm joints: mean=%.4f median=%.4f min=%.4f"%(maxdiff.mean(),np.median(maxdiff),maxdiff.min()))
print("fraction of chunks with maxdiff > 0.0087 rad (reject threshold): %.1f%%"%(100*(maxdiff>thr).mean()))
print("per-joint fraction over threshold:", np.round((np.abs(A[:,0,:14]-S[:,:14])>thr).mean(axis=0),3))

# simulate the degrade path used in core.py for chunk_size(8) <= K(40)
def degrade(chunk, state):
    ch = torch.tensor(chunk, dtype=torch.float64).clone()
    orig = ch.clone()
    n = ch.shape[0]
    for i in range(14):
        if abs(float(orig[0,i]) - float(state[i])) > thr:
            ch[:,i] = cubic_hermite_segment(float(state[i]), float(orig[-1,i]), n,
                                            start_velocity=0.0, end_velocity=0.0, dtype=ch.dtype)
    return ch

exe=[]
for k in range(A.shape[0]):
    ch = degrade(A[k], S[k])
    ch,_ = remove_boundary_rollbacks(ch, list(S[k,:14]), joint_count=14, window_size=10)
    ch,_ = remove_small_rollbacks(ch, joint_count=14, window_size=10, max_rollback_steps=2)
    ch,_ = remove_open_gripper_loops(ch, joint_count=14, joints_per_arm=7, left_gripper_index=14,
        right_gripper_index=15, min_excursion=math.radians(1.0), max_excursion=math.radians(8.0),
        max_return_gap=math.radians(0.6), max_return_ratio=0.2, max_duration_steps=30,
        open_gripper_threshold=0.1, gripper_margin_steps=3, continuation_steps=3)
    ch = smooth_action_chunk(ch, joint_count=14, passes=1)
    ch,_ = smooth_large_excursions(ch, joint_count=14, wave_threshold=math.radians(100.0))
    exe.append(ch.numpy())
E=np.stack(exe)

def prof(X, cols=slice(0,14)):
    sp=np.abs(np.diff(X[:,:,cols],axis=1)).sum(axis=2)
    m=sp.mean(axis=1,keepdims=True); m[m==0]=1
    return (sp/m).mean(axis=0)
print("\nintra-chunk normalized speed profile, arm joints:")
print("  raw 200k model output :", np.round(prof(A),3), " peak/edge=%.2f"%(prof(A).max()/((prof(A)[0]+prof(A)[-1])/2)))
p=prof(E)
print("  after deploy pipeline :", np.round(p,3), " peak/edge=%.2f"%(p.max()/((p[0]+p[-1])/2)))
print("  (100k recorded, filtered, from report: 0.634 0.903 1.187 1.311 1.286 0.997 0.684  peak/edge=1.99)")

def stats(X):
    d=np.diff(X[:,:,:14],axis=1)
    intra=np.abs(d).mean()
    seam=np.abs(X[1:,0,:14]-X[:-1,-1,:14]).mean()
    x=d.reshape(-1,14)
    ac=np.mean([np.corrcoef(X[k,:-1,j],X[k,1:,j])[0,1] if np.std(X[k,:-1,j])>0 and np.std(X[k,1:,j])>0 else np.nan
                for k in range(X.shape[0]) for j in range(14)])
    dd=np.diff(X[:,:,:14],axis=1)
    rev=np.mean([(np.sign(dd[k,1:,j])*np.sign(dd[k,:-1,j])<0).mean() for k in range(X.shape[0]) for j in range(14)])
    return intra,seam,rev
for name,X in (("raw model",A),("executed (simulated)",E)):
    i,s,r=stats(X)
    print(f"\n{name}: intra|d|={i:.5f} seam|jump|={s:.5f} seam/intra={s/i:.2f} reversal={100*r:.1f}%")
