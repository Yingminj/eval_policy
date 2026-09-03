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
 '+fail36 (250k)': J+'act_tidy_up_stationery_le_batch_success_361_fail_36_2026-08-18_05-11-55-952726/run/checkpoints/250000/pretrained_model',
 '+fail72 (250k)': J+'act_tidy_up_stationery_le_batch_success_361_fail_72_2026-08-18_06-09-55-723522/run/checkpoints/250000/pretrained_model',
}
H=100; dt={'action':[i/30 for i in range(H)]}
SETS={
 'success (seen)': (R+'batch_success_361_fail_72', list(range(72,435,9))),
 'fail (seen by +36/+72)': (R+'batch_success_361_fail_72', list(range(0,72,2))),
 'batch_1 unseen 08-07': (R+'batch_1', list(range(0,63,2))),
}
DS={k:LeRobotDataset('local/x',root=v[0],episodes=v[1],delta_timestamps=dt) for k,v in SETS.items()}
rng=np.random.default_rng(0)
IDX={k:rng.choice(len(d),min(600,len(d)),replace=False) for k,d in DS.items()}

def run(pol,pre,post,ds,idx):
    errs=[]
    with torch.no_grad():
        for k in range(0,len(idx),16):
            batch=[ds[int(i)] for i in idx[k:k+16]]
            b={key:torch.stack([x[key] for x in batch]).cuda() for key in batch[0] if isinstance(batch[0][key],torch.Tensor)}
            b['task']=[x['task'] for x in batch]
            gt=b['action'].clone()
            pb=pre(dict(b))
            pred=pol.predict_action_chunk(pb)
            pred=post(pred)
            errs.append((pred[:,:H].to(gt.device)-gt).abs().mean(dim=(1,2)).cpu().numpy())
    return np.concatenate(errs)

res={}
for name,p in CK.items():
    pol=ACTPolicy.from_pretrained(p).cuda().eval()
    pre,post=make_pre_post_processors(pol.config, pretrained_path=p)
    pre=make_pre_post_processors(pol.config, pretrained_path=p, preprocessor_overrides={'device_processor':{'device':'cuda'}})[0]; post=make_pre_post_processors(pol.config, pretrained_path=p)[1]
    res[name]={sn: float(run(pol,pre,post,DS[sn],IDX[sn]).mean()) for sn in SETS}
    del pol; torch.cuda.empty_cache()
    print(name, {k:round(v,4) for k,v in res[name].items()}, flush=True)
print()
hdr=list(SETS)
print('%-16s'%'open-loop chunk L1 (rad)'[:16] + ''.join('%-24s'%h for h in hdr))
for n in CK: print('%-16s'%n + ''.join('%-24s'%('%.4f'%res[n][h]) for h in hdr))
json.dump(res,open(sys.argv[1],'w'),indent=1)
