import numpy as np
A=np.load("/home/kewei/.claude/jobs/b715062d/tmp/A200k.npy")
def binom(A,jc=14,passes=1):
    out=A.copy()
    for _ in range(passes):
        n=out.copy()
        n[:,1:-1,:jc]=(0.25*out[:,:-2,:jc]+0.5*out[:,1:-1,:jc]+0.25*out[:,2:,:jc])
        out=n
    return out
def stats(X,label):
    d=np.diff(X,axis=1)
    fl=(np.sign(d[:,1:,:])!=np.sign(d[:,:-1,:]))&(np.abs(d[:,1:,:])>1e-4)
    ac=[]
    for c in range(d.shape[0]):
        for j in range(d.shape[2]):
            x=d[c,:,j]-d[c,:,j].mean()
            if np.std(x)>1e-9: ac.append(np.corrcoef(x[:-1],x[1:])[0,1])
    path=np.abs(d).sum(axis=1); net=np.abs(X[:,-1,:]-X[:,0,:]); k=net>1e-4
    print(f"{label:<46} |d|={np.abs(d).mean():.5f} ac={np.nanmean(ac):+.3f} pathnet={np.median(path[k]/net[k]):.2f} rev={fl.mean()*100:.1f}%")
print("=== filter contribution measured DIRECTLY on 200k raw arms (0-13) ===")
stats(A[:,:,:14],"200k raw arms")
stats(binom(A)[:,:,:14],"200k raw arms + binomial filter")
print("\n(08-13 100k FILTERED arms measured: ac=+0.448 rev=4.7% pathnet=1.00)")
