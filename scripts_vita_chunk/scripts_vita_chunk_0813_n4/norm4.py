import numpy as np
P="/home/kewei/YING/paper/policy/scripts_vita_chunk_0813pm"
T="/home/kewei/.claude/jobs/b715062d/tmp"
A8,S8=np.load(f"{P}/A200k.npy"),np.load(f"{P}/S200k.npy")
A4,S4=np.load(f"{T}/A200k_n4.npy"),np.load(f"{T}/S200k_n4.npy")
for A,S,lab in ((A8,S8,"n=8"),(A4,S4,"n=4 NEW")):
    dS=np.diff(S[:,:14],axis=0)
    print(f"{lab:<9} chunks={A.shape[0]:>4} steps/chunk={A.shape[1]} total_actions={A.shape[0]*A.shape[1]:>4}")
    print(f"          state travel/chunk arms = {np.abs(dS).sum()/len(dS)/14:.5f} rad   "
          f"total arm travel = {np.abs(dS).sum():.3f} rad   net displacement = {np.abs(S[-1,:14]-S[0,:14]).sum():.3f}")
    # per-chunk net motion of the commanded trajectory
    net=np.abs(A[:,-1,:14]-A[:,0,:14]).mean()
    intra=np.abs(np.diff(A[:,:,:14],axis=1)).mean()
    print(f"          per-chunk arm net motion = {net:.5f}   intra|step| = {intra:.5f}   "
          f"jitter/motion = {intra/(net/(A.shape[1]-1)):.3f}  (1.0 = perfectly monotone)")
    # state velocity
    print(f"          state |vel| per chunk-interval = {np.abs(dS).mean():.5f}")
# common-scale comparison: normalise step size by measured state speed
print("\n--- scale-free within-chunk metrics on ARMS ---")
def sf(X,lab):
    d=np.diff(X[:,:,:14],axis=1)
    path=np.abs(d).sum(axis=1); net=np.abs(X[:,-1,:14]-X[:,0,:14])
    k=net>1e-4
    print(f"{lab:<22} path/net mean={np.mean(path[k]/net[k]):.3f} p90={np.percentile(path[k]/net[k],90):.3f} "
          f"frac_chunks_nonmonotone={(np.mean((path[k]/net[k])>1.05))*100:.1f}%")
sf(A8[:,:4],"n=8 [first 4]"); sf(A4,"n=4 NEW")
