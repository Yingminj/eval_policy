import glob, hashlib, json
import numpy as np, pandas as pd
def fps(root):
    out={}
    for f in sorted(glob.glob(root+"/data/**/*.parquet",recursive=True)):
        df=pd.read_parquet(f,columns=["episode_index","action","observation.state"])
        for ep,g in df.groupby("episode_index"):
            a=np.stack(g["action"].values).astype(np.float32)
            out.setdefault(int(ep),[]).append(a)
    return {k:hashlib.md5(np.concatenate(v).tobytes()).hexdigest() for k,v in out.items()}, \
           {k:np.concatenate(v) for k,v in out.items()}
TR="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361"
EV="/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data"
ht,at=fps(TR); he,ae=fps(EV)
print("train eps",len(ht),"eval eps",len(he))
exact=set(ht.values())&set(he.values())
print("EXACT duplicate episodes:",len(exact))
# near-duplicate: same length + tiny distance
tl={k:v.shape[0] for k,v in at.items()}
near=0; hits=[]
for ke,ve in ae.items():
    cands=[k for k,l in tl.items() if l==ve.shape[0]]
    for kt in cands:
        d=np.abs(at[kt]-ve).mean()
        if d<1e-4: near+=1; hits.append((ke,kt,float(d))); break
print("NEAR-duplicate episodes:",near)
print("examples",hits[:5])
json.dump({"exact":len(exact),"near":near,"train_eps":len(ht),"eval_eps":len(he),
           "hits":[(int(a),int(b),c) for a,b,c in hits]},
          open("/tmp/claude-1000/-home-kewei-YING-robot-data-platform/38c37621-e06a-4aff-9fb5-e41444b2918b/scratchpad/contam.json","w"),indent=1)
