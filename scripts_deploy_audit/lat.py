import time, torch, numpy as np
from lerobot.policies.factory import make_policy_config
from lerobot.policies.vita.modeling_vita import VitaPolicy
p = VitaPolicy.from_pretrained("/mnt/robot_platform/jobs/vita_tidy_up_stationery_le_batch_3_2026-08-12_12-06-10-620381/run/checkpoints/200000/pretrained_model")
p.eval().to("cuda")
b = {"observation.state": torch.zeros(1,1,16,device="cuda"),
     "observation.images": torch.zeros(1,1,3,3,480,640,device="cuda")}
with torch.inference_mode():
    for _ in range(3): p.vita.generate_actions(b)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(20): p.vita.generate_actions(b)
    torch.cuda.synchronize()
print("VITA generate_actions: %.1f ms/call"%((time.perf_counter()-t)/20*1000))
