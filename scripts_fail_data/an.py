import pandas as pd, numpy as np, glob, json
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def load(d):
    fs=sorted(glob.glob(R+d+'/data/*/*.parquet'))
    df=pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return df
for d in ['batch_success_361','batch_fail_72']:
    df=load(d)
    print('==',d, df.shape, list(df.columns))
    A=np.stack(df['action'].values); S=np.stack(df['observation.state'].values)
    print(' action mean', np.round(A.mean(0),3))
    print(' action std ', np.round(A.std(0),3))
    print(' state  mean', np.round(S.mean(0),3))
    print(' state  std ', np.round(S.std(0),3))
    # per-episode motion
    ep=df['episode_index'].values
    d1=np.abs(np.diff(A,axis=0)).sum(1); d1=np.concatenate([[0],d1])
    same=np.concatenate([[False],ep[1:]==ep[:-1]])
    mv=d1[same]
    print(' |dA| per step: mean %.4f med %.4f  frac<1e-3: %.3f'%(mv.mean(),np.median(mv),(mv<1e-3).mean()))
    # gripper
    print(' gripper L range', A[:,14].min(), A[:,14].max(), ' R', A[:,15].min(), A[:,15].max())
