import math, numpy as np, torch
from traj_standalone import (cubic_hermite_segment, remove_boundary_rollbacks,
    remove_small_rollbacks, remove_open_gripper_loops, smooth_action_chunk, smooth_large_excursions)
SP="/home/kewei/YING/paper/policy/scripts_vita_chunk/scripts_vita_chunk_0813pm/"
A=np.load(SP+"A200k.npy"); S=np.load(SP+"S200k.npy"); thr=0.0087
def prof(X):
    sp=np.abs(np.diff(X[:,:,:14],axis=1)).sum(axis=2); m=sp.mean(axis=1,keepdims=True); m[m==0]=1
    p=(sp/m).mean(axis=0); return p, p.max()/((p[0]+p[-1])/2)
def herm(chunk,state):
    ch=torch.tensor(chunk,dtype=torch.float64).clone(); o=ch.clone(); n=ch.shape[0]
    for i in range(14):
        if abs(float(o[0,i])-float(state[i]))>thr:
            ch[:,i]=cubic_hermite_segment(float(state[i]),float(o[-1,i]),n,start_velocity=0.0,end_velocity=0.0,dtype=ch.dtype)
    return ch
H=np.stack([herm(A[k],S[k]).numpy() for k in range(len(A))])
B=np.stack([smooth_action_chunk(torch.tensor(A[k],dtype=torch.float64),joint_count=14,passes=1).numpy() for k in range(len(A))])
for n,X in (("raw",A),("binomial smoothing only",B),("Hermite degrade only",H)):
    p,r=prof(X); print(f"{n:26s}", np.round(p,3), "peak/edge=%.2f"%r)
# net displacement retained
net_raw=np.abs(A[:,-1,:14]-A[:,0,:14]).sum(axis=1)
net_exe=np.abs(H[:,-1,:14]-H[:,0,:14]).sum(axis=1)
print("\nnet |displacement| per chunk (14 arm joints, rad): raw model %.4f  executed %.4f"%(net_raw.mean(),net_exe.mean()))
print("path length per chunk: raw %.4f  hermite %.4f"%(np.abs(np.diff(A[:,:,:14],axis=1)).sum(axis=(1,2)).mean(),
                                                       np.abs(np.diff(H[:,:,:14],axis=1)).sum(axis=(1,2)).mean()))
# start/end step speed of executed chunk
sp=np.abs(np.diff(H[:,:,:14],axis=1)).sum(axis=2)
print("executed per-step speed: first %.5f  mid %.5f  last %.5f rad/step"%(sp[:,0].mean(),sp[:,3].mean(),sp[:,-1].mean()))
# n4
SP4="/home/kewei/YING/paper/policy/scripts_vita_chunk/scripts_vita_chunk_0813_n4/"
A4=np.load(SP4+"A200k_n4.npy"); S4=np.load(SP4+"S200k_n4.npy")
m4=np.abs(A4[:,0,:14]-S4[:,:14]).max(axis=1)
print("\nn=4 recording: chunks=%d  max|a0-state| median=%.4f  frac>thr=%.1f%%"%(len(A4),np.median(m4),100*(m4>thr).mean()))
