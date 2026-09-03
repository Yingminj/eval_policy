import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2")
sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2/detr")

from detr.main import get_args_parser
from detr.models import build_ACT_model

H, W = 480, 640


def run(bb):
    p = argparse.ArgumentParser(parents=[get_args_parser()])
    a = p.parse_args([])
    a.backbone, a.camera_names = bb, ["top", "wrist_L", "wrist_R"]
    a.hidden_dim, a.dim_feedforward, a.nheads = 512, 3200, 8
    a.enc_layers, a.dec_layers, a.num_queries, a.action_dim = 4, 7, 100, 16
    m = build_ACT_model(a).cuda().eval()
    img = torch.randn(1, 3, 3, H, W, device="cuda")
    qpos = torch.randn(1, 16, device="cuda")
    with torch.no_grad():
        for _ in range(8):
            m(qpos, img, None)
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            m(qpos, img, None)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    r = dict(
        backbone=bb,
        params_total=sum(x.numel() for x in m.parameters() if x.requires_grad),
        params_backbone=sum(x.numel() for x in m.backbones.parameters() if x.requires_grad),
        infer_ms_p50=ts[15],
    )
    del m
    torch.cuda.empty_cache()
    return r


for bb in ["resnet18", "resnet34", "resnet50"]:
    print(json.dumps(run(bb)), flush=True)
