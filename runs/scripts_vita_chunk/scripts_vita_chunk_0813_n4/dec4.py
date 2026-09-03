import numpy as np
P="/home/kewei/YING/paper/policy/scripts_vita_chunk_0813pm"; T="/home/kewei/.claude/jobs/b715062d/tmp"
A8,S8=np.load(f"{P}/A200k.npy"),np.load(f"{P}/S200k.npy")
A4,S4=np.load(f"{T}/A200k_n4.npy"),np.load(f"{T}/S200k_n4.npy")
print("=== seam share of total commanded path (arms) ===")
for A,lab,L in ((A8,"n=8",8),(A4,"n=4 NEW",4)):
    st=np.abs(np.diff(A.reshape(-1,16)[:,:14],axis=0)).sum(axis=1)
    ph=np.array([st[i::L].mean() for i in range(L)])
    print(f"{lab:<9} seam phase={ph[-1]:.4f}  intra phases mean={ph[:-1].mean():.4f}  "
          f"seam share of path = {ph[-1]/ph.sum()*100:.1f}%   seam/intra = {ph[-1]/ph[:-1].mean():.2f}x")

print("\n=== whole-recording excess path (arms): total |step| / |net displacement| ===")
for A,lab in ((A8,"n=8"),(A4,"n=4 NEW")):
    X=A.reshape(-1,16)[:,:14]
    path=np.abs(np.diff(X,axis=0)).sum(axis=0); net=np.abs(X[-1]-X[0])
    k=net>1e-3
    print(f"{lab:<9} median path/net = {np.median(path[k]/net[k]):.2f}   mean = {(path[k]/net[k]).mean():.2f}  "
          f"(joints counted {k.sum()}/14)")

print("\n=== gripper state range (OOD check, training bound = [0,1]) ===")
for S,lab in ((S8,"n=8"),(S4,"n=4 NEW")):
    print(f"{lab:<9} left  [{S[:,14].min():.4f}, {S[:,14].max():.4f}]   right [{S[:,15].min():.4f}, {S[:,15].max():.4f}]")
for A,lab in ((A8,"n=8"),(A4,"n=4 NEW")):
    print(f"{lab:<9} action left [{A[:,:,14].min():.4f}, {A[:,:,14].max():.4f}] right [{A[:,:,15].min():.4f}, {A[:,:,15].max():.4f}]")

print("\n=== per-frame progress ===")
for A,S,lab,L in ((A8,S8,"n=8",8),(A4,S4,"n=4 NEW",4)):
    dS=np.abs(np.diff(S[:,:14],axis=0)).sum(axis=1)
    print(f"{lab:<9} arm state travel per frame = {dS.sum()/((len(S)-1)*L):.5f} rad   "
          f"frames={A.shape[0]*L}  total travel={dS.sum():.2f}  net={np.abs(S[-1,:14]-S[0,:14]).sum():.2f}  "
          f"efficiency net/travel={np.abs(S[-1,:14]-S[0,:14]).sum()/dS.sum():.3f}")

print("\n=== does the chunk actually move? per-chunk commanded net vs state lag ===")
for A,S,lab in ((A8,S8,"n=8"),(A4,S4,"n=4 NEW")):
    net=np.abs(A[:,-1,:14]-A[:,0,:14]).sum(axis=1)
    lag=np.abs(A[:,0,:14]-S[:,:14]).sum(axis=1)
    print(f"{lab:<9} commanded net/chunk={net.mean():.4f}  |a0-state|/chunk={lag.mean():.4f}  ratio lag/net={lag.mean()/net.mean():.2f}")
