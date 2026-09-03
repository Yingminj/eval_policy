import numpy as np
SP="/tmp/claude-1000/-home-kewei-YING-robot-data-platform/3f79e71e-0bc0-4e0e-9ea6-e3c85a7c5058/scratchpad"
OLD="/home/kewei/YING/paper/policy/scripts_vita_chunk"
A_new=np.load(f"{SP}/A100k.npy")          # 100k ckpt, WITH filters
A_old=np.load(f"{OLD}/A.npy")             # 10k ckpt, NO filters

def binom(A, joint_count=14, passes=1):
    """Replicate smooth_action_chunk: binomial [.25,.5,.25], endpoints + cols>=14 fixed."""
    out=A.copy()
    for _ in range(passes):
        nxt=out.copy()
        nxt[:,1:-1,:joint_count]=(0.25*out[:,:-2,:joint_count]
                                  +0.5*out[:,1:-1,:joint_count]
                                  +0.25*out[:,2:,:joint_count])
        out=nxt
    return out

def stats(A,label,cols=None):
    sl=slice(None) if cols is None else cols
    X=A[:,:,sl]
    d=np.diff(X,axis=1)
    intra=np.abs(d)
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
    print(f"{label:<44} |d|={intra.mean():.5f}  ac={np.nanmean(ac):+.3f}  "
          f"pathnet={np.median(path[keep]/net[keep]):.2f}  rev={flips.mean()*100:.1f}%  "
          f"n/s={np.mean(ra)/np.mean(ta):.2f}")

ARM=slice(0,14); GRIP=slice(14,16)
print("=== ALL 16 joints ===")
stats(A_old,        "yesterday 10k, raw (as reported)")
stats(binom(A_old), "yesterday 10k, + today's binomial filter")
stats(A_new,        "today 100k, as recorded (filtered)")

print("\n=== ARM joints 0-13 only (filter DOES touch these) ===")
stats(A_old,"yesterday 10k raw",ARM); stats(binom(A_old),"yesterday 10k + filter",ARM); stats(A_new,"today 100k (filtered)",ARM)

print("\n=== GRIPPERS 14-15 only (filter NEVER touches these = raw model) ===")
stats(A_old,"yesterday 10k raw grippers",GRIP)
stats(A_new,"today 100k raw grippers",GRIP)
