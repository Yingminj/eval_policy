import pandas as pd, numpy as np, glob
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def load(d):
    fs=sorted(glob.glob(R+d+'/data/*/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return (np.stack(df['observation.state'].values).astype(np.float32),
            np.stack(df['action'].values).astype(np.float32),
            df['episode_index'].values)
Ss,As,es=load('batch_success_361'); sd=Ss.std(0)+1e-6
H=100; EPS=0.15; GAP=45   # 0.15 sigma, >1.5 s apart
def revisit(S,A,ep,name,maxep=80):
    fr=[]; div=[]
    ids=np.unique(ep)[:maxep]
    for e in ids:
        m=ep==e; s=S[m]/sd; a=A[m]
        n=len(s)
        if n<H+GAP: continue
        D=np.sqrt(((s[:,None,:]-s[None,:,:])**2).sum(-1))
        iu=np.triu_indices(n,GAP)
        hit=D[iu]<EPS
        fr.append(hit.mean())
        i0,j0=iu[0][hit],iu[1][hit]
        ok=(j0+H<=n)
        if ok.sum():
            i0,j0=i0[ok],j0[ok]
            div.append(np.abs(a[i0[:,None]+np.arange(H)]-a[j0[:,None]+np.arange(H)]).mean())
    print('%-22s revisit-pair rate %.4f  | action-chunk MAE at revisits %.4f rad (n_ep=%d)'%(
        name, np.mean(fr), np.mean(div) if div else float('nan'), len(fr)))
revisit(Ss,As,es,'success')
Sf,Af,ef=load('batch_fail_72'); revisit(Sf,Af,ef,'fail_72')
