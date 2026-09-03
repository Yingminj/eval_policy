import pandas as pd, numpy as np, glob
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def load(d):
    fs=sorted(glob.glob(R+d+'/data/*/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return np.stack(df['action'].values).astype(np.float32), df['episode_index'].values
for d in ['batch_success_361','batch_fail_72']:
    A,ep=load(d)
    rows=[]
    for e in np.unique(ep):
        a=A[ep==e]
        ev=[]
        for gi in (14,15):
            b=(a[:,gi]>0.5).astype(np.int8)
            idx=np.flatnonzero(np.diff(b))
            ev.extend(idx.tolist())
        ev=sorted(ev)
        if len(ev)!=4: continue
        rows.append([ev[0], ev[1]-ev[0], ev[2]-ev[1], ev[3]-ev[2], len(a)-ev[3], len(a)])
    r=np.array(rows)
    print('==',d,'n',len(r))
    print('   pre-grasp1  g1..g2   g2..g3   g3..g4   post   total (median frames)')
    print('   ', np.round(np.median(r,0),0))
    print('    mean       ', np.round(r.mean(0),0))
