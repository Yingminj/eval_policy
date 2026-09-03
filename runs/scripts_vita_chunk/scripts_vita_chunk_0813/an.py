import ast, numpy as np
SP="/tmp/claude-1000/-home-kewei-YING-robot-data-platform/3f79e71e-0bc0-4e0e-9ea6-e3c85a7c5058/scratchpad"
TXT="/home/kewei/YING/robot_data_platform/record_chunk.txt"
JOINTS=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
KEYS=[f"{j}.pos" for j in JOINTS]
states,chunks,cur=[],[],None
for line in open(TXT):
    line=line.strip()
    if line.startswith("=== Chunk"):
        if cur: chunks.append(cur)
        cur=[]
    elif line.startswith("Robot State"):
        states.append(ast.literal_eval(line.split(":",1)[1].strip()))
    elif line.startswith("Action "):
        d=ast.literal_eval(line.split(":",1)[1].strip()); cur.append([d[k] for k in KEYS])
if cur: chunks.append(cur)
A=np.array(chunks); S=np.array(states)
np.save(f"{SP}/A100k.npy",A); np.save(f"{SP}/S100k.npy",S)
print("actions:",A.shape,"states:",S.shape)

intra=np.abs(np.diff(A,axis=1)); seam=np.abs(A[1:,0,:]-A[:-1,-1,:])
print("\n--- magnitude (rad) ---")
print(f"intra |step delta| mean={intra.mean():.5f} p95={np.percentile(intra,95):.5f} max={intra.max():.5f}")
print(f"seam  |jump|       mean={seam.mean():.5f} p95={np.percentile(seam,95):.5f} max={seam.max():.5f}")
print(f"seam/intra = {seam.mean()/intra.mean():.2f}x")
d0=np.abs(A[:,0,:]-S); print(f"|action0-state| mean={d0.mean():.6f} max={d0.max():.6f}")

d=np.diff(A,axis=1)
flips=(np.sign(d[:,1:,:])!=np.sign(d[:,:-1,:]))&(np.abs(d[:,1:,:])>1e-4)
print(f"reversal rate: {flips.mean()*100:.1f}%")

path=np.abs(np.diff(A,axis=1)).sum(axis=1); net=np.abs(A[:,-1,:]-A[:,0,:])
print(f"path/net all-joint median={np.median(path/np.maximum(net,1e-9)):.2f}")
keep=net>1e-4; r=path[keep]/net[keep]
print(f"path/net moving-only median={np.median(r):.2f} p90={np.percentile(r,90):.2f}")

ac=[]
for c in range(d.shape[0]):
    for j in range(d.shape[2]):
        x=d[c,:,j]-d[c,:,j].mean()
        if np.std(x)>1e-9: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
print(f"lag-1 autocorr of step deltas: mean={np.nanmean(ac):.3f}")

t=np.arange(A.shape[1]); ra,ta=[],[]; R=np.empty_like(A)
for c in range(A.shape[0]):
    for j in range(A.shape[2]):
        y=A[c,:,j]; k,b=np.polyfit(t,y,1); fit=k*t+b
        R[c,:,j]=y-fit; ra.append(np.abs(y-fit).mean()); ta.append(np.abs(fit[-1]-fit[0]))
ra=np.array(ra); ta=np.array(ta)
print(f"trend span mean={ta.mean():.5f}  resid mean={ra.mean():.5f}  noise/signal={ra.mean()/ta.mean():.2f}")

flat=R.reshape(-1,16); C=np.corrcoef(flat.T); iu=np.triu_indices(16,1); v=C[iu]
print(f"cross-joint resid corr: mean={v.mean():+.3f} mean|r|={np.abs(v).mean():.3f} frac|r|>0.5={np.mean(np.abs(v)>0.5)*100:.0f}%")

sd=np.diff(A.reshape(-1,16),axis=0)
print(f"executed stream |step delta| mean={np.abs(sd).mean():.5f} p99={np.percentile(np.abs(sd),99):.5f} max={np.abs(sd).max():.5f}")
print("\nmean |delta| by position in chunk:")
for i in range(d.shape[1]): print(f"  {i}->{i+1}: {np.abs(d[:,i,:]).mean():.5f}")

print("\n--- per joint ---")
print(f"{'joint':<22}{'intra_mean':>11}{'intra_max':>11}{'seam_mean':>11}{'seam_max':>11}{'rev%':>8}{'resid':>9}{'trend':>9}")
for i,j in enumerate(JOINTS):
    rr=np.abs(R[:,:,i]).mean()
    tt=np.abs(A[:,-1,i]-A[:,0,i]).mean()
    print(f"{j:<22}{intra[:,:,i].mean():>11.5f}{intra[:,:,i].max():>11.5f}{seam[:,i].mean():>11.5f}{seam[:,i].max():>11.5f}{flips[:,:,i].mean()*100:>7.1f}%{rr:>9.5f}{tt:>9.5f}")
