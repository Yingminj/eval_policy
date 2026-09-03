import pandas as pd, numpy as np, glob
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def load(d):
    fs=sorted(glob.glob(R+d+'/data/*/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return np.stack(df['action'].values).astype(np.float32), df['episode_index'].values
for d in ['batch_success_361','batch_fail_72']:
    A,ep=load(d)
    counts=[];lens=[]
    for e in np.unique(ep):
        m=ep==e; a=A[m]
        n=0
        for g in (a[:,14],a[:,15]):
            b=(g>0.5).astype(np.int8)
            n+=int(np.abs(np.diff(b)).sum())
        counts.append(n); lens.append(m.sum())
    counts=np.array(counts); lens=np.array(lens)
    print('==',d,'n_ep',len(counts))
    print('  gripper toggles/ep: mean %.2f med %.1f p90 %.1f'%(counts.mean(),np.median(counts),np.percentile(counts,90)))
    print('  ep len: mean %.0f med %.0f'%(lens.mean(),np.median(lens)))
    print('  toggle hist', np.bincount(counts)[:16])
