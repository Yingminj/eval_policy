import ast, numpy as np
SP="/home/kewei/.claude/jobs/b715062d/tmp"
JOINTS=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
KEYS=[f"{j}.pos" for j in JOINTS]
def parse(p):
    states,chunks,cur=[],[],None
    for line in open(p):
        line=line.strip()
        if line.startswith("=== Chunk"):
            if cur is not None: chunks.append(cur)
            cur=[]
        elif line.startswith("Robot State"):
            states.append(ast.literal_eval(line.split(":",1)[1].strip()))
        elif line.startswith("Action "):
            d=ast.literal_eval(line.split(":",1)[1].strip()); cur.append([d[k] for k in KEYS])
    if cur is not None: chunks.append(cur)
    return np.array(chunks),np.array(states)
A,S=parse("/home/kewei/YING/robot_data_platform/record_chunk.txt")
np.save(f"{SP}/A200k_n4.npy",A); np.save(f"{SP}/S200k_n4.npy",S)
print("actions:",A.shape,"states:",S.shape)

def stats(X,label):
    d=np.diff(X,axis=1); intra=np.abs(d)
    flips=(np.sign(d[:,1:,:])!=np.sign(d[:,:-1,:]))&(np.abs(d[:,1:,:])>1e-4)
    ac=[]
    for c in range(d.shape[0]):
        for j in range(d.shape[2]):
            x=d[c,:,j]-d[c,:,j].mean()
            if np.std(x)>1e-9 and len(x)>2: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
    path=np.abs(d).sum(axis=1); net=np.abs(X[:,-1,:]-X[:,0,:]); keep=net>1e-4
    t=np.arange(X.shape[1]); ra,ta=[],[]
    for c in range(X.shape[0]):
        for j in range(X.shape[2]):
            y=X[c,:,j]; k,b=np.polyfit(t,y,1); fit=k*t+b
            ra.append(np.abs(y-fit).mean()); ta.append(np.abs(fit[-1]-fit[0]))
    print(f"{label:<34} |d|={intra.mean():.5f} ac={np.nanmean(ac):+.3f} "
          f"pathnet={np.median(path[keep]/net[keep]):.2f} rev={flips.mean()*100:.1f}% n/s={np.mean(ra)/np.mean(ta):.2f}")

A200=np.load("/home/kewei/YING/paper/policy/scripts_vita_chunk_0813pm/A200k.npy")  # 200k RAW n=8
print("\n=== ALL 16 ===");   stats(A200,"200k RAW n_action=8"); stats(A,"200k RAW n_action=4 (NEW)")
print("\n=== ARMS 0-13 ===");stats(A200[:,:,:14],"200k RAW n=8"); stats(A[:,:,:14],"200k RAW n=4 (NEW)")
print("\n=== GRIPPERS ===");  stats(A200[:,:,14:],"200k RAW n=8"); stats(A[:,:,14:],"200k RAW n=4 (NEW)")
# n=8 truncated to first 4 steps => apples-to-apples on window length
print("\n=== n=8 recording, first 4 steps only (same window length) ===")
stats(A200[:,:4,:],"200k n=8 [0:4] ALL"); stats(A200[:,:4,:14],"200k n=8 [0:4] ARMS")

def seamrep(X,S_,label):
    intra=np.abs(np.diff(X,axis=1)); seam=np.abs(X[1:,0,:]-X[:-1,-1,:])
    print(f"{label:<22} intra={intra.mean():.5f} seam={seam.mean():.5f} ratio={seam.mean()/intra.mean():.2f}x "
          f"seam_p95={np.percentile(seam,95):.5f} |a0-state|={np.abs(X[:,0,:]-S_).mean():.5f} "
          f"arms|a0-s|={np.abs(X[:,0,:14]-S_[:,:14]).mean():.5f}")
print("\n=== seam ===")
seamrep(A200,np.load("/home/kewei/YING/paper/policy/scripts_vita_chunk_0813pm/S200k.npy"),"n=8")
seamrep(A,S,"n=4 (NEW)")

# continuous stream jitter: what the robot actually sees
for X,lab,L in ((A200,"n=8",8),(A,"n=4 (NEW)",4)):
    st=np.abs(np.diff(X.reshape(-1,16),axis=0))
    arms=np.abs(np.diff(X.reshape(-1,16)[:,:14],axis=0))
    ph=np.array([st[i::L].mean() for i in range(L)])
    print(f"\n{lab} continuous stream: mean|step|(all)={st.mean():.5f} arms={arms.mean():.5f}")
    print("  phase speed: "+" ".join(f"{v:.4f}" for v in ph)+f"   modulation={ph.max()/ph.min():.2f}x (last=seam)")
    d=np.diff(X.reshape(-1,16)[:,:14],axis=0); ac=[]
    for j in range(14):
        x=d[:,j]-d[:,j].mean()
        if np.std(x)>1e-9: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
    rev=((np.sign(d[1:])!=np.sign(d[:-1]))&(np.abs(d[1:])>1e-4)).mean()*100
    print(f"  arms continuous: ac={np.mean(ac):+.3f} rev={rev:.1f}%")
