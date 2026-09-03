"""Sweep enc_layers / dec_layers for the ACT_rosbag2 DETRVAE model.

Reports: trainable params (split by module), inference latency, train-step latency,
peak CUDA memory. Config mirrors configs/apex_real_C3_task.yaml.
"""
import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2")
sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2/detr")

from detr.main import get_args_parser  # noqa: E402
from detr.models import build_ACT_model  # noqa: E402

BASE = dict(
    lr=1e-5,
    lr_backbone=1e-5,
    backbone="resnet18",
    camera_names=["top", "wrist_L", "wrist_R"],
    hidden_dim=512,
    dim_feedforward=3200,
    nheads=8,
    num_queries=100,
    dropout=0.1,
    action_dim=16,
)

H, W = 480, 640
BS_TRAIN = 8


def count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build(enc_layers, dec_layers, vae_enc_layers=None):
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args([])
    for k, v in BASE.items():
        setattr(args, k, v)
    args.enc_layers = enc_layers
    args.dec_layers = dec_layers
    model = build_ACT_model(args)
    if vae_enc_layers is not None and vae_enc_layers != enc_layers:
        # rebuild only the CVAE style encoder with a different depth
        import copy

        from detr.models.transformer import TransformerEncoder, TransformerEncoderLayer

        layer = TransformerEncoderLayer(512, 8, 3200, 0.1, "relu", False)
        model.encoder = TransformerEncoder(copy.deepcopy(layer), vae_enc_layers, None)
    return model.cuda()


@torch.no_grad()
def bench_infer(model, reps=30):
    model.eval()
    img = torch.randn(1, 3, 3, H, W, device="cuda")
    qpos = torch.randn(1, 16, device="cuda")
    for _ in range(8):
        model(qpos, img, None)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(qpos, img, None)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.95)]


def bench_train(model, reps=12):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    img = torch.randn(BS_TRAIN, 3, 3, H, W, device="cuda")
    qpos = torch.randn(BS_TRAIN, 16, device="cuda")
    act = torch.randn(BS_TRAIN, 100, 16, device="cuda")
    is_pad = torch.zeros(BS_TRAIN, 100, dtype=torch.bool, device="cuda")

    def step():
        a_hat, _, (mu, logvar) = model(qpos, img, None, act, is_pad)
        loss = torch.nn.functional.l1_loss(a_hat, act)
        loss = loss + 10.0 * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps * 1e3
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dt, peak


def run(enc, dec, vae=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(enc, dec, vae)
    row = dict(
        enc_layers=enc,
        dec_layers=dec,
        vae_enc_layers=vae if vae is not None else enc,
        params_total=count(m),
        params_backbone=count(m.backbones),
        params_vae_encoder=count(m.encoder),
        params_obs_encoder=count(m.transformer.encoder),
        params_decoder=count(m.transformer.decoder),
    )
    row["infer_ms_p50"], row["infer_ms_p95"] = bench_infer(m)
    row["train_ms_per_step"], row["train_peak_gb"] = bench_train(m)
    del m
    torch.cuda.empty_cache()
    return row


if __name__ == "__main__":
    grid = []
    # depth sweeps around the repo default (enc=4, dec=7)
    for dec in [1, 2, 4, 7, 10]:
        grid.append((4, dec, None))
    for enc in [1, 2, 6, 8]:
        grid.append((enc, 7, None))
    # decouple VAE encoder depth from obs encoder depth (lerobot-style)
    grid.append((4, 1, 4))   # lerobot default shape
    grid.append((4, 7, 1))
    rows = []
    for enc, dec, vae in grid:
        r = run(enc, dec, vae)
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(
        "/tmp/claude-1000/-home-kewei-YING-paper/ed6a271b-9f98-40a6-b2f0-0affbdf2f743/scratchpad/act_layer_sweep.json",
        "w",
    ) as f:
        json.dump(rows, f, indent=2)
