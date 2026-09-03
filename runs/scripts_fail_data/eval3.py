import torch, numpy as np, json, sys
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
J='/mnt/robot_platform/jobs/'
CK={
 'succ361 (200k)': J+'act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model',
 '+fail36 (200k)': J+'act_tidy_up_stationery_le_batch_success_361_fail_36_2026-08-18_05-11-55-952726/run/checkpoints/200000/pretrained_model',
 '+fail72 (200k)': J+'act_tidy_up_stationery_le_batch_success_361_fail_72_2026-08-18_06-09-55-723522/run/checkpoints/200000/pretrained_model',
}
H=100; dt={'action':[i/30 for i in range(H)]}
SETS={
 'success':        (R+'batch_success_361_fail_72', list(range(72,435,12))),
 'fail seen@36':   (R+'batch_success_361_fail_72', list(range(0,36,2))),
 'fail unseen@36': (R+'batch_success_361_fail_72', list(range(36,72,2))),
}
DS={k:LeRobotDataset('local/x',root=v[0],episodes=v[1],delta_timestamps=dt) for k,v in SETS.items()}
rng=np.random.default_rng(0)
IDX={k:rng.choice(len(d),min(400,len(d)),replace=False) for k,d in DS.items()}
MASK_ROWS=120   # top 25% of 480 = background above the table

def run(pol,pre,post,ds,idx,mask=False):
    errs=[]
    with torch.no_grad():
        for k in range(0,len(idx),16):
            batch=[ds[int(i)] for i in idx[k:k+16]]
            b={key:torch.stack([x[key] for x in batch]).cuda() for key in batch[0] if isinstance(batch[0][key],torch.Tensor)}
            b['task']=[x['task'] for x in batch]
            if mask:
                b['observation.images.top']=b['observation.images.top'].clone()
                b['observation.images.top'][:,:,:MASK_ROWS,:]=0.5
            gt=b['action'].clone()
            pred=post(pol.predict_action_chunk(pre(dict(b))))
            errs.append((pred[:,:H].to(gt.device)-gt).abs().mean(dim=(1,2)).cpu().numpy())
    return float(np.concatenate(errs).mean())

res={}
for name,p in CK.items():
    pol=ACTPolicy.from_pretrained(p).cuda().eval()
    pre=make_pre_post_processors(pol.config,pretrained_path=p,preprocessor_overrides={'device_processor':{'device':'cuda'}})[0]
    post=make_pre_post_processors(pol.config,pretrained_path=p)[1]
    r={}
    for sn in SETS:
        r[sn]=run(pol,pre,post,DS[sn],IDX[sn])
        r[sn+' [bg masked]']=run(pol,pre,post,DS[sn],IDX[sn],mask=True)
    res[name]=r; print(name, {k:round(v,4) for k,v in r.items()}, flush=True)
    del pol; torch.cuda.empty_cache()
json.dump(res,open(sys.argv[1],'w'),indent=1)
