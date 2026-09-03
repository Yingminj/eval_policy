import pandas as pd, numpy as np, glob
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
rng=np.random.default_rng(0)
def load(d):
    fs=sorted(glob.glob(R+d+'/data/*/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    S=np.stack(df['observation.state'].values).astype(np.float32)
    A=np.stack(df['action'].values).astype(np.float32)
    ep=df['episode_index'].values
    return S,A,ep
Ss,As,es=load('batch_success_361')
Sf,Af,ef=load('batch_fail_72')
H=100  # ACT chunk size
sd=Ss.std(0)+1e-6

def chunks(S,A,ep,idx):
    out=[]
    keep=[]
    for i in idx:
        j=i+H
        if j<len(A) and ep[j-1]==ep[i]:
            out.append(A[i:j]); keep.append(i)
    return np.array(out), np.array(keep)

# sample query points
qi=rng.choice(len(Sf)-H, 3000, replace=False)
Cf,qi=chunks(Sf,Af,ef,qi)
Q=Sf[qi]/sd

# build success index (subsample for speed)
si=rng.choice(len(Ss)-H, 60000, replace=False)
Cs,si=chunks(Ss,As,es,si)
K=Ss[si]/sd

def nn_stats(Q,Cq,K,Cs,label,exclude_same_ep=None):
    ds=[]
    for b in range(0,len(Q),256):
        d=((Q[b:b+256,None,:]-K[None,:,:])**2).sum(-1)
        ds.append(d)
    D=np.concatenate(ds)
    nn=D.argmin(1); dmin=np.sqrt(D.min(1))
    diff=np.abs(Cq-Cs[nn]).mean(axis=(1,2))
    return dmin, diff

d_fs, c_fs = nn_stats(Q,Cf,K,Cs,'fail->succ')
# baseline: success query -> success neighbor (different episode)
qi2=rng.choice(len(Ss)-H, 3000, replace=False)
Cs2,qi2=chunks(Ss,As,es,qi2)
Q2=Ss[qi2]/sd
d_ss, c_ss = nn_stats(Q2,Cs2,K,Cs,'succ->succ')

print('state-NN distance (normalized L2, 16-dim):')
print('  fail query  -> nearest success frame: med %.3f  p90 %.3f'%(np.median(d_fs),np.percentile(d_fs,90)))
print('  succ query  -> nearest success frame: med %.3f  p90 %.3f'%(np.median(d_ss),np.percentile(d_ss,90)))
print('future-%d-step action chunk MAE vs that neighbor (rad):'%H)
print('  fail vs success: med %.4f  mean %.4f'%(np.median(c_fs),c_fs.mean()))
print('  succ vs success: med %.4f  mean %.4f'%(np.median(c_ss),c_ss.mean()))
# restrict to close matches only
for thr in [0.5,1.0]:
    m1=d_fs<thr; m2=d_ss<thr
    if m1.sum()>50 and m2.sum()>50:
        print('  [d<%.1f] fail-vs-succ %.4f (n=%d) | succ-vs-succ %.4f (n=%d)'%(
            thr, np.median(c_fs[m1]), m1.sum(), np.median(c_ss[m2]), m2.sum()))
