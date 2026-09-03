import json, time, torch
from lerobot.policies.patch_policy.modeling_patch_policy import PatchPolicy
J="/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-"
out={}
for tag,c in [("A_act",J+"21-58-821581"),("B_diffusion",J+"31-19-689756")]:
    p=PatchPolicy.from_pretrained(c+"/run/checkpoints/200000/pretrained_model").to("cuda").eval()
    b={"observation.images":torch.rand(1,5,3,3,480,640,device="cuda"),
       "observation.state":torch.zeros(1,5,16,device="cuda")}
    with torch.no_grad():
        for _ in range(3): p.model.predict(b)
        torch.cuda.synchronize()
        ts=[]
        for _ in range(20):
            t=time.perf_counter(); p.model.predict(b); torch.cuda.synchronize()
            ts.append(time.perf_counter()-t)
    ts=sorted(ts); k=p.config.n_action_steps
    out[tag]={"median_s":round(ts[10],4),"p10_s":round(ts[2],4),"p90_s":round(ts[18],4),
              "n_action_steps":k,"own_window_s":round(k/30,3),
              "duty_own":round(ts[10]/(k/30),3),"duty_deployed_8":round(ts[10]/(8/30),2),
              "denoise_steps":getattr(p.model,"num_inference_steps",None)}
    print(tag,out[tag],flush=True)
    del p; torch.cuda.empty_cache()
json.dump(out,open("/tmp/claude-1000/-home-kewei-YING-robot-data-platform/38c37621-e06a-4aff-9fb5-e41444b2918b/scratchpad/latency.json","w"),indent=1)
