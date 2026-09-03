import ast, numpy as np
SP="/home/kewei/.claude/jobs/b715062d/tmp"
JOINTS=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
KEYS=[f"{j}.pos" for j in JOINTS]
def parse(p):
    states,chunks,cur=[],[],None
    for line in open(p):
        line=line.strip()
        if line.startswith("=== Chunk"):
            if cur: chunks.append(cur)
            cur=[]
        elif line.startswith("Robot State"):
            states.append(ast.literal_eval(line.split(":",1)[1].strip()))
        elif line.startswith("Action "):
            d=ast.literal_eval(line.split(":",1)[1].strip()); cur.append([d[k] for k in KEYS])
    if cur: chunks.append(cur)
    return np.array(chunks),np.array(states)
A,S=parse("/home/kewei/YING/robot_data_platform/record_chunk.txt")
np.save(f"{SP}/A200k.npy",A); np.save(f"{SP}/S200k.npy",S)
print("actions:",A.shape,"states:",S.shape)

def stats(X,label):
    d=np.diff(X,axis=1); intra=np.abs(d)
    flips=(np.sign(d[:,1:,:])!=np.sign(d[:,:-1,:]))&(np.abs(d[:,1:,:])>1e-4)
    ac=[]
    for c in range(d.shape[0]):
        for j in range(d.shape[2]):
            x=d[c,:,j]-d[c,:,j].mean()
            if np.std(x)>1e-9: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
    path=np.abs(d).sum(axis=1); net=np.abs(X[:,-1,:]-X[:,0,:]); keep=net>1e-4
    t=np.arange(X.shape[1]); ra,ta=[],[]
    for c in range(X.shape[0]):
        for j in range(X.shape[2]):
            y=X[c,:,j]; k,b=np.polyfit(t,y,1); fit=k*t+b
            ra.append(np.abs(y-fit).mean()); ta.append(np.abs(fit[-1]-fit[0]))
    print(f"{label:<40} |d|={intra.mean():.5f} ac={np.nanmean(ac):+.3f} "
          f"pathnet={np.median(path[keep]/net[keep]):.2f}(p90 {np.percentile(path[keep]/net[keep],90):.2f}) "
          f"rev={flips.mean()*100:.1f}% n/s={np.mean(ra)/np.mean(ta):.2f}")

A10=np.load("/home/kewei/YING/paper/policy/scripts_vita_chunk/A.npy")           # 10k RAW
A100=np.load("/home/kewei/YING/paper/policy/scripts_vita_chunk_0813/A100k.npy") # 100k FILTERED
print("\n=== ALL 16 joints ===")
stats(A10,"10k  RAW      (08-12)"); stats(A100,"100k FILTERED (08-13)"); stats(A,"200k RAW      (08-13 pm)")
print("\n=== ARM joints 0-13 ===")
stats(A10[:,:,:14],"10k  RAW"); stats(A100[:,:,:14],"100k FILTERED"); stats(A[:,:,:14],"200k RAW")
print("\n=== GRIPPERS 14-15 (raw in all three) ===")
stats(A10[:,:,14:],"10k  RAW"); stats(A100[:,:,14:],"100k RAW"); stats(A[:,:,14:],"200k RAW")

intra=np.abs(np.diff(A,axis=1)); seam=np.abs(A[1:,0,:]-A[:-1,-1,:])
print(f"\n=== magnitude ===\nintra mean={intra.mean():.5f} p95={np.percentile(intra,95):.5f} max={intra.max():.5f}")
print(f"seam  mean={seam.mean():.5f} p95={np.percentile(seam,95):.5f} max={seam.max():.5f}  seam/intra={seam.mean()/intra.mean():.2f}x")
d0=np.abs(A[:,0,:]-S); ex=d0<1e-6
print(f"|action0-state| mean={d0.mean():.5f} max={d0.max():.5f}   exact-match={ex.mean()*100:.1f}% (should be ~0 now)")
v_prev=A[:-1,-1,:]-A[:-1,-2,:]; sm=A[1:,0,:]-A[:-1,-1,:]; v_next=A[1:,1,:]-A[1:,0,:]
m=np.abs(v_prev)>1e-4
print(f"\n=== seam direction ===\nseam agrees w/ prev vel: {np.mean(np.sign(sm[m])==np.sign(v_prev[m]))*100:.1f}%   "
      f"prev vs next vel same sign: {np.mean(np.sign(v_prev[m])==np.sign(v_next[m]))*100:.1f}%")
print(f"|seam|/|v_prev| median={np.median(np.abs(sm[m])/np.abs(v_prev[m])):.2f}")
print(f"|state(k+1)-last_action(k)| mean={np.abs(S[1:]-A[:-1,-1,:]).mean():.5f} max={np.abs(S[1:]-A[:-1,-1,:]).max():.5f}")
sp=np.abs(np.diff(A,axis=1)).sum(axis=2); spn=sp/np.maximum(sp.mean(axis=1,keepdims=True),1e-9)
print("\n=== in-chunk speed profile (train ref ~1.0 flat) ===")
print("  "+" ".join(f"{v:.3f}" for v in spn.mean(axis=0))+f"   peak/edge={spn.mean(axis=0)[3]/((spn.mean(axis=0)[0]+spn.mean(axis=0)[6])/2):.2f}")
st=np.abs(np.diff(A.reshape(-1,16),axis=0)).sum(axis=1)
ph=np.array([st[i::8].mean() for i in range(8)])
print("phase speed: "+" ".join(f"{v:.4f}" for v in ph)+f"   modulation={ph.max()/ph.min():.2f}x  (phase7=seam)")
print("\n=== per joint ===")
print(f"{'joint':<22}{'intra_mean':>11}{'intra_max':>11}{'seam_mean':>11}{'seam_max':>11}{'rev%':>8}")
fl=(np.sign(np.diff(A,axis=1)[:,1:,:])!=np.sign(np.diff(A,axis=1)[:,:-1,:]))&(np.abs(np.diff(A,axis=1)[:,1:,:])>1e-4)
for i,j in enumerate(JOINTS):
    print(f"{j:<22}{intra[:,:,i].mean():>11.5f}{intra[:,:,i].max():>11.5f}{seam[:,i].mean():>11.5f}{seam[:,i].max():>11.5f}{fl[:,:,i].mean()*100:>7.1f}%")
