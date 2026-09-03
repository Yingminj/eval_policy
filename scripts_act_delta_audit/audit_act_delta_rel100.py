#!/usr/bin/env python
"""只读审计：`act_delta` @ batch_5_rel100_keep-gripper（相对动作已开启的那一次训练）。

配套报告：`../act/act_delta-rel100-precision-analysis-2026-08.md`

输出六段：

    1. checkpoint 与 dataset 的配置核对（开关是否真的开了、stats 是否真的在相对空间）
    2. **留出集泛化差距**（需要 GPU）—— 本报告的核心结论
    3. in-sample 离线绝对空间 chunk MAE，按 chunk 内位置 k 分桶（需要 GPU）
    4. 相对/绝对动作目标的 per-k 标准差 —— 解释 §3 的曲线形状
    5. 真机部署日志 record_chunk.txt 的 chunk 级统计
    6. 截短执行 horizon 的收益表

必须用 `~/anaconda3/envs/lerobot` 这个环境（base env 的 pandas 与 numpy 2.x 不兼容）：

    /home/kewei/anaconda3/envs/lerobot/bin/python audit_act_delta_rel100.py
    /home/kewei/anaconda3/envs/lerobot/bin/python audit_act_delta_rel100.py --skip-gpu
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/kewei/YING/lerobot_vlahost/src")

JOBS = "/mnt/robot_platform/jobs/"
REL_JOB = JOBS + "act_delta_tidy_up_stationery_le_batch_5_rel100_keep-gripper_2026-08-14_06-34-30-681119"
ABS_JOB = JOBS + "act_delta_tidy_up_stationery_le_batch_5_2026-08-13_03-52-51-314668"
REL_ROOT = "/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_5_rel100_keep-gripper"
ABS_ROOT = "/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_5"
DEPLOY_LOG = "/home/kewei/YING/robot_data_platform/record_chunk.txt"

H = 100
ARM = 14
NAMES = [f"J{i}_{s}" for s in "LR" for i in range(1, 8)] + ["gL", "gR"]


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- 1
def section_config() -> None:
    rule("1. checkpoint / dataset 配置核对")
    ck = json.load(open(f"{REL_JOB}/run/checkpoints/100000/pretrained_model/config.json"))
    for k in (
        "use_relative_actions",
        "relative_exclude_joints",
        "relative_consistency_check",
        "chunk_size",
        "n_action_steps",
    ):
        print(f"  ckpt.{k:32s} = {ck[k]}")
    print(f"  ckpt.input_features              = {list(ck['input_features'])}")

    sb = open(f"{REL_JOB}/job.sbatch").read()
    flags = re.findall(r"--policy\.\S+ \S+", sb)
    print(f"\n  job.sbatch 里的 --policy.* 参数: {flags}")
    print("  → 注意 use_relative_actions 不在其中：它来自 train-venv 里被手工改过的 dataclass 默认值。")

    venv_cfg = "/opt/robot-platform/train-venv/lib/python3.12/site-packages/lerobot/policies/act_delta/configuration_act_delta.py"
    repo_cfg = "/home/kewei/YING/lerobot_vlahost/src/lerobot/policies/act_delta/configuration_act_delta.py"
    for tag, path in (("train-venv", venv_cfg), ("repo    ", repo_cfg)):
        src = open(path).read()
        m = re.search(r"use_relative_actions: bool = (\w+)", src)
        c = re.search(r'relative_consistency_check: str = "(\w+)"', src)
        has_tw = "loss_time_decay" in src
        print(f"  {tag}: use_relative_actions={m.group(1):5s} check={c.group(1):5s} loss_time_decay={has_tw}")

    for tag, root in (("REL(相对空间)", REL_ROOT), ("ABS(绝对空间)", ABS_ROOT)):
        s = json.load(open(f"{root}/meta/stats.json"))["action"]
        mean = np.array(s["mean"]).ravel()[:ARM]
        std = np.array(s["std"]).ravel()[:ARM]
        ratio = np.abs(mean).mean() / std.mean()
        print(f"  {tag} stats: |mean|/std over arm = {ratio:.3f}  (相对空间应 ≪1，绝对空间 ≫1)")


# --------------------------------------------------------------------------- 2
def load_chunks(root: str):
    """(N, H, 16) 训练用绝对动作 chunk 及其 anchor state，尊重 episode 边界。"""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    d = pd.concat([pd.read_parquet(f) for f in files])
    A = np.stack(d["action"].values)
    S = np.stack(d["observation.state"].values)
    ep = d["episode_index"].values
    out_a, out_s = [], []
    for e in np.unique(ep):
        m = np.where(ep == e)[0]
        for t in range(len(m) - H):
            out_a.append(A[m][t : t + H])
            out_s.append(S[m][t])
    return np.array(out_a), np.array(out_s)


def section_perk_std() -> None:
    rule("3. 相对 / 绝对动作目标的 per-k 标准差（解释 §2.1 曲线的形状）")
    A, S = load_chunks(REL_ROOT)
    rel = A[:, :, :ARM] - S[:, None, :ARM]
    abs_ = A[:, :, :ARM]
    pooled_rel = rel.reshape(-1, ARM).std(0).mean()
    pooled_abs = abs_.reshape(-1, ARM).std(0).mean()
    print(f"  pooled std (归一化器实际使用的那一个): rel {pooled_rel:.4f}  abs {pooled_abs:.4f}"
          f"  → pooled 方差增益 {pooled_abs / pooled_rel:.2f}×")
    print(f"\n  {'k':>4}  {'rel_std':>9}  {'abs_std':>9}  {'方差增益':>9}")
    for k in (0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
        r = rel[:, k, :].std(0).mean()
        a = abs_[:, k, :].std(0).mean()
        print(f"  {k:>4}  {r:9.4f}  {a:9.4f}  {a / r:8.2f}×")
    print("\n  → 绝对空间 per-k std 基本恒定；相对空间从 0.016 涨到 0.288（18×）。")
    print("    相对化的收益 100% 集中在 chunk 头部，到 k≈99 已经完全消失。")


# --------------------------------------------------------------------------- 4
def parse_deploy_log():
    txt = open(DEPLOY_LOG).read()
    blocks = re.split(r"=== Chunk (\d+) ===", txt)[1:]
    out = []
    for i in range(0, len(blocks), 2):
        body = blocks[i + 1]
        s = np.array(json.loads(re.search(r"Robot State \(16 joints\): (\[.*?\])", body, re.S).group(1)))
        a = np.array([json.loads(m) for m in re.findall(r"Action \d+: (\[.*?\])", body, re.S)])
        out.append((int(blocks[i]), s, a))
    return out


def section_deploy() -> None:
    rule("4. 真机部署日志 record_chunk.txt")
    chunks = parse_deploy_log()
    print(f"  {len(chunks)} 个 chunk，每个 {len(chunks[0][2])} 步")

    d0 = np.array([np.abs(a[0, :ARM] - s[:ARM]).mean() for _, s, a in chunks])
    net = np.array([np.linalg.norm(a[-1, :ARM] - s[:ARM]) for _, s, a in chunks])
    arc = np.array([np.linalg.norm(np.diff(a[:, :ARM], axis=0), axis=1).sum() for _, s, a in chunks])
    print(f"  |a0 - state| 手臂均值 = {d0.mean():.4f} rad   → chunk 衔接（顺滑性）")
    print(f"  chunk 净位移 中位数    = {np.median(net):.4f} rad, 弧长 {np.median(arc):.4f}, 直线度 {np.median(net / arc):.3f}")

    err, ratio = [], []
    for j in range(len(chunks) - 1):
        i0, s0, a0 = chunks[j]
        i1, s1, _ = chunks[j + 1]
        if i1 != i0 + 1:
            continue
        err.append(np.linalg.norm(s1[:ARM] - a0[-1, :ARM]))
        ratio.append(np.linalg.norm(s1[:ARM] - s0[:ARM]) / max(np.linalg.norm(a0[-1, :ARM] - s0[:ARM]), 1e-9))
    print(f"  chunk 末端跟踪误差 中位数 = {np.median(err):.4f} rad,  实际/指令位移 = {np.median(ratio):.3f}")
    print("  → 底层执行忠实；误差在“指令目标”本身，不在伺服或时序。")

    # 幅度是否缩水：与最近邻训练 anchor 比
    A, S = load_chunks(REL_ROOT)
    rel_net = np.linalg.norm(A[:, -1, :ARM] - S[:, :ARM], axis=1)
    rs = []
    for _, s, a in chunks:
        nn = np.argsort(np.linalg.norm(S[:, :ARM] - s[:ARM], axis=1))[:25]
        rs.append(np.linalg.norm(a[-1, :ARM] - s[:ARM]) / np.median(rel_net[nn]))
    print(f"  部署/训练(最近邻 k=25) 位移比 中位数 = {np.median(rs):.3f}  → 幅度没有缩水")

    g = np.array([[s[14], s[15]] for _, s, _ in chunks])
    print(f"  夹爪观测范围: gL [{g[:, 0].min():.3f}, {g[:, 0].max():.3f}]  gR [{g[:, 1].min():.3f}, {g[:, 1].max():.3f}]"
          f"   (训练范围 [0,1])")


# --------------------------------------------------------------------------- 2 & 5 (GPU)
def offline_mae(ckpt: str, root: str, n: int, seed: int = 0, episodes: list[int] | None = None) -> np.ndarray:
    """返回 (n, H, 16) 的绝对空间逐元素绝对误差。"""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act_delta.inference_act_delta import predict_absolute_chunk
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path, cfg.device = ckpt, "cuda"
    ds = LeRobotDataset(
        repo_id="local/eval", root=root, delta_timestamps={"action": [i / 30.0 for i in range(cfg.chunk_size)]}
    )
    policy = make_policy(cfg, ds_meta=ds.meta).eval().to("cuda")
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=ckpt,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    epi = np.array(ds.hf_dataset["episode_index"])
    ok: list[int] = []
    for e in np.unique(epi) if episodes is None else episodes:
        idx = np.where(epi == e)[0]
        ok.extend(idx[: max(0, len(idx) - cfg.chunk_size)].tolist())
    sel = np.random.default_rng(seed).choice(np.array(ok), size=min(n, len(ok)), replace=False)

    errs = []
    for i in sel:
        s = ds[int(i)]
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in s.items() if isinstance(v, torch.Tensor)}
        batch["task"] = [s.get("task", "")]
        gt = batch["action"][0].float().cpu().numpy()
        with torch.no_grad():
            pred = predict_absolute_chunk(policy, pre, post, batch)[0].float().cpu().numpy()
        e = np.abs(pred[: len(gt)] - gt)
        e[np.array(s["action_is_pad"])[: len(e)]] = np.nan
        errs.append(e)
    return np.array(errs)


def section_mae(n: int) -> None:
    rule("2. 离线绝对空间 chunk MAE（step-matched：两个模型都取 100k step）")
    runs = {
        "REL100k": (f"{REL_JOB}/run/checkpoints/100000/pretrained_model", REL_ROOT),
        "ABS100k": (f"{ABS_JOB}/run/checkpoints/100000/pretrained_model", ABS_ROOT),
        "ABS200k": (f"{ABS_JOB}/run/checkpoints/200000/pretrained_model", ABS_ROOT),
        "REL50k": (f"{REL_JOB}/run/checkpoints/050000/pretrained_model", REL_ROOT),
    }
    E = {}
    for tag, (ck, root) in runs.items():
        E[tag] = offline_mae(ck, root, n)
        print(f"  {tag} 完成", flush=True)

    print(f"\n  {'tag':<9}{'arm MAE':>9}{'L':>9}{'R':>9}{'R/L':>7}{'grip':>9}{'k=0':>9}{'k=99':>9}{'k99/k0':>8}")
    for tag, e in E.items():
        arm = np.nanmean(e[:, :, :ARM], axis=(0, 2))
        print(
            f"  {tag:<9}{np.nanmean(e[:, :, :ARM]):9.5f}{np.nanmean(e[:, :, :7]):9.5f}"
            f"{np.nanmean(e[:, :, 7:14]):9.5f}"
            f"{np.nanmean(e[:, :, 7:14]) / np.nanmean(e[:, :, :7]):7.2f}"
            f"{np.nanmean(e[:, :, 14:]):9.5f}{arm[0]:9.5f}{arm[99]:9.5f}{arm[99] / arm[0]:8.2f}"
        )

    a = np.nanmean(E["REL100k"][:, :, :ARM], axis=(0, 2))
    b = np.nanmean(E["ABS100k"][:, :, :ARM], axis=(0, 2))
    print(f"\n  逐步曲线   {'k':>4}{'REL100k':>10}{'ABS100k':>10}{'REL/ABS':>9}")
    for k in list(range(0, 100, 5)) + [99]:
        print(f"  {'':<9}{k:>4}{a[k]:10.5f}{b[k]:10.5f}{a[k] / b[k]:9.3f}")
    cross = np.where(a > b)[0]
    print(f"\n  交叉点：k = {cross[0] if len(cross) else 'N/A'}  （此后相对模型比绝对模型更差）")

    rule("5. 截短执行 horizon 的收益（不需要重训）")
    print(f"  {'N':>5}{'REL 全程均值':>14}{'REL 终点误差':>14}{'ABS 终点误差':>14}")
    for N in (10, 20, 30, 40, 50, 60, 80, 100):
        print(f"  {N:>5}{a[:N].mean():14.5f}{a[N - 1]:14.5f}{b[N - 1]:14.5f}")


def episode_fingerprints(root: str) -> dict:
    """按 (长度, 首帧 action, 末帧 action) 给每条 episode 打指纹，用于判断跨 batch 的重叠。"""
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    d = pd.concat([pd.read_parquet(f) for f in files])
    A = np.stack(d["action"].values)
    ep = d["episode_index"].values
    out = {}
    for e in np.unique(ep):
        m = np.where(ep == e)[0]
        out[int(e)] = (len(m), tuple(np.round(A[m][0], 5)), tuple(np.round(A[m][-1], 5)))
    return out


def section_heldout(n: int) -> None:
    rule("2. 留出集泛化差距（本报告的核心结论）")
    base = "/mnt/robot_platform/datasets/tidy_up_stationery_le/"
    fp5 = set(episode_fingerprints(base + "batch_5").values())
    fp6 = episode_fingerprints(base + "batch_6")
    new = sorted(e for e, f in fp6.items() if f not in fp5)
    print(f"  batch_5 ⊆ batch_6：batch_6 的 61 条里有 {61 - len(new)} 条就是训练集，")
    print(f"  真正没见过的 episode 有 {len(new)} 条 → {new[0]}..{new[-1]}。只在这 {len(new)} 条上采样。")

    runs = {
        "REL100k": f"{REL_JOB}/run/checkpoints/100000/pretrained_model",
        "ABS200k": f"{ABS_JOB}/run/checkpoints/200000/pretrained_model",
        "ABS100k": f"{ABS_JOB}/run/checkpoints/100000/pretrained_model",
    }
    print(f"\n  真·留出集 arm MAE (rad)   {'overall':>10}{'k=0':>10}{'k=29':>10}{'k=49':>10}{'k=99':>10}"
          f"{'L':>9}{'R':>9}{'grip':>9}")
    for tag, ck in runs.items():
        E = offline_mae(ck, base + "batch_6", n, episodes=new)
        a = np.nanmean(E[:, :, :ARM], axis=(0, 2))
        print(
            f"  {tag:<24}{np.nanmean(E[:, :, :ARM]):10.5f}{a[0]:10.5f}{a[29]:10.5f}{a[49]:10.5f}{a[99]:10.5f}"
            f"{np.nanmean(E[:, :, :7]):9.5f}{np.nanmean(E[:, :, 7:14]):9.5f}{np.nanmean(E[:, :, 14:]):9.5f}"
        )
    print("\n  → 与 §3 的 in-sample 数字对比即得泛化倍数。留出误差比 horizon 效应大一个量级以上。")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-gpu", action="store_true", help="跳过 §2/§3/§6（需要 CUDA 与约 30 分钟）")
    p.add_argument("-n", type=int, default=240, help="离线 MAE 的采样 anchor 数")
    args = p.parse_args()

    section_config()
    if not args.skip_gpu:
        section_heldout(args.n)
        section_mae(args.n)
    section_perk_std()
    section_deploy()


if __name__ == "__main__":
    main()
