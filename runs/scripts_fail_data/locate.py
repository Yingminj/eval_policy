import pandas as pd,glob,numpy as np
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def load(d):
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(R+d+'/data/*/*.parquet'))],ignore_index=True)
    A=np.stack(df['action'].values).astype(np.float32); ep=df['episode_index'].values
    sig={}
    for e in np.unique(ep):
        a=A[ep==e]; sig[e]=(len(a), tuple(np.round(a[0],4)))
    return sig
m=load('batch_success_361_fail_36'); f=load('batch_fail_36')
fs=set(f.values())
idx=sorted([e for e,s in m.items() if s in fs])
print('n fail eps located in merged:',len(idx))
print('indices:',idx)
np.save('/tmp/fail_idx.npy',np.array(idx))
