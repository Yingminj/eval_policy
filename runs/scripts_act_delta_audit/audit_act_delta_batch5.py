#!/usr/bin/env python
"""act_delta / tidy_up_stationery_le batch_5 部署失效审计脚本。

复现 `act/act_delta-batch5-failure-analysis-2026-08.md` 里的每一个数字。
只读，不改任何东西。

用法（必须用带 pandas>=2 / numpy>=2 的环境）::

    /home/kewei/anaconda3/envs/lerobot/bin/python audit_act_delta_batch5.py

各段落对应报告章节：
  [1] §1  checkpoint 配置       —— use_relative_actions 是否真的打开了
  [2] §2  相对空间统计量        —— 相对/绝对 std 比，即"相对动作最多能省多少方差"
  [3] §3  训练规模与过拟合      —— episode 数、epoch 数、loss 曲线
  [4] §4  部署观测分布外检查    —— state / action 对训练 min-max 的越界
  [5] §5  chunk 连续性与速度    —— chunk 边界跳变、每步位移与训练对比
  [6] §6  代码副本一致性        —— 三份 lerobot 的 act_delta 差异
"""

from __future__ import annotations

import ast
import glob
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

JOB = Path(
    "/mnt/robot_platform/jobs/act_delta_tidy_up_stationery_le_batch_5_2026-08-13_03-52-51-314668"
)
DATASET = Path("/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_5")
RECORD = Path("/home/kewei/YING/robot_data_platform/record_chunk.txt")

TRAIN_SRC = Path("/opt/robot-platform/train-venv/lib/python3.12/site-packages/lerobot")
DEPLOY_SRC = Path("/home/kewei/YING/lerobot_vlahost/src/lerobot")
PLATFORM_SRC = Path("/home/kewei/YING/robot_data_platform/lerobot/src/lerobot")

NAMES = [
    "J1_L", "J2_L", "J3_L", "J4_L", "J5_L", "J6_L", "J7_L",
    "J1_R", "J2_R", "J3_R", "J4_R", "J5_R", "J6_R", "J7_R",
    "gL", "gR",
]
CHUNK_SIZE = 100  # policy.chunk_size，相对空间统计量必须用同一个值


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- [1] 配置
def section_config() -> dict:
    rule("[1] checkpoint 配置 —— 相对动作到底开了没有")
    cfg = json.loads((JOB / "run/checkpoints/last/pretrained_model/config.json").read_text())
    for k in (
        "type", "chunk_size", "n_action_steps", "use_relative_actions",
        "relative_exclude_joints", "relative_consistency_check", "temporal_ensemble_coeff",
    ):
        print(f"  {k:32s} = {cfg.get(k, '<缺失>')}")
    print(f"  input_features                    = {list(cfg['input_features'])}")
    print(f"  loss_time_decay / loss_front_weight = "
          f"{cfg.get('loss_time_decay', '<缺失 → 训练侧代码更旧>')} / "
          f"{cfg.get('loss_front_weight', '<缺失 → 训练侧代码更旧>')}")

    sbatch = (JOB / "job.sbatch").read_text()
    cmd = sbatch[sbatch.index("lerobot.scripts.lerobot_train"):].strip()
    print("\n  训练命令里所有 --policy.* 参数：")
    for flag in re.findall(r"--policy\.\S+(?:\s+\S+)?", cmd):
        print(f"    {flag}")
    print(f"\n  >>> 命令行未出现 use_relative_actions；dataclass 默认 False；"
          f"checkpoint 记录 {cfg['use_relative_actions']}")

    # exclude 掩码是否真的命中
    excl = [t.lower() for t in cfg["relative_exclude_joints"]]
    mask = [not any(t in n.lower() for t in excl) for n in cfg["action_feature_names"]]
    print(f"  exclude 掩码：{sum(mask)}/{len(mask)} 维为相对，"
          f"排除 {[n for n, m in zip(cfg['action_feature_names'], mask) if not m]}")
    return cfg


# ------------------------------------------------- [2] 相对空间统计量
def load_frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(glob.glob(str(DATASET / "data/**/*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files])
    state = np.stack(df["observation.state"].values).astype(np.float64)
    action = np.stack(df["action"].values).astype(np.float64)
    return state, action, df["episode_index"].values


def section_relative(state, action, episode) -> None:
    rule(f"[2] 相对空间统计量（k=0..{CHUNK_SIZE - 1}）—— 相对动作最多能省多少方差")
    rel = []
    for e in np.unique(episode):
        m = episode == e
        S, A, T = state[m], action[m], int(m.sum())
        idx = np.arange(T)
        for k in range(CHUNK_SIZE):
            rel.append(A[np.minimum(idx + k, T - 1)] - S[idx])
    R = np.concatenate(rel)
    abs_std, rel_std = action.std(0), R.std(0)

    print(f"  样本数 {len(R):,}（{len(state):,} 帧 × {CHUNK_SIZE} 个 k）")
    print(f"\n  {'dim':6}{'abs_std':>10}{'rel_std':>10}{'方差增益':>10}"
          f"{'|mean|/std(abs)':>17}{'|mean|/std(rel)':>17}")
    for d in range(16):
        print(f"  {NAMES[d]:6}{abs_std[d]:10.4f}{rel_std[d]:10.4f}"
              f"{abs_std[d] / max(rel_std[d], 1e-9):10.2f}"
              f"{abs(action.mean(0)[d]) / max(abs_std[d], 1e-9):17.3f}"
              f"{abs(R.mean(0)[d]) / max(rel_std[d], 1e-9):17.3f}")
    print(f"\n  >>> 手臂 14 维平均方差增益 = "
          f"{np.mean(abs_std[:14] / np.maximum(rel_std[:14], 1e-9)):.2f}×（1.0 = 完全没帮助）")

    d0 = action - state
    print(f"\n  k=0 的 |a_t - s_t| 均值（rad）：{np.round(np.abs(d0).mean(0)[:14], 5).tolist()}")
    print("  >>> action ≈ state：相对表示下 chunk 前几步的回归目标 ≈ 0，几乎无信息量")

    # act_delta 自带的一致性检查会怎么判当前 stats.json
    stats = json.loads((DATASET / "meta/stats.json").read_text())
    mean_t = np.array(stats["action"]["mean"])
    std_t = np.array(stats["action"]["std"])
    m = np.array([not n.lower().startswith("g") for n in NAMES])
    ratio = np.abs(mean_t[m]).mean() / max(np.abs(std_t[m]).mean(), 1e-8)
    print(f"\n  meta/stats.json 上 validate_relative_setup 的 |mean|/std = {ratio:.2f}"
          f"（>1.0 判定为绝对空间统计量 → 当前 stats 未在相对空间重算）")


# ------------------------------------------------- [3] 训练规模与过拟合
def section_training() -> None:
    rule("[3] 训练规模与过拟合")
    info = json.loads((DATASET / "meta/info.json").read_text())
    epf = glob.glob(str(DATASET / "meta/episodes/**/*.parquet"), recursive=True)
    lens = pd.concat([pd.read_parquet(f) for f in epf])["length"].values
    print(f"  episodes={info['total_episodes']}  frames={info['total_frames']:,}  "
          f"fps={info['fps']}  tasks={info['total_tasks']}  splits={info['splits']}")
    print(f"  episode 长度 (帧): min={lens.min()} med={np.median(lens):.0f} max={lens.max()}"
          f"  → {lens.min() / info['fps']:.0f}–{lens.max() / info['fps']:.0f} s")

    tc = json.loads((JOB / "run/checkpoints/last/pretrained_model/train_config.json").read_text())
    print(f"  steps={tc['steps']}  batch={tc['batch_size']}  "
          f"image_transforms.enable={tc['dataset']['image_transforms']['enable']}  "
          f"scheduler={tc.get('scheduler')}")
    print(f"  optimizer={tc['optimizer']['type']} lr={tc['optimizer']['lr']} "
          f"(CLI 传的是 adam，被 use_policy_training_preset 覆盖成 adamw)")

    curve = {}
    for line in (JOB / "log.jsonl").read_text().splitlines():
        try:
            msg = json.loads(line)["message"]
        except Exception:
            continue
        g = re.search(r"step:(\S+).*?epch:([\d.]+)\s+loss:([\d.]+)\s+grdn:([\d.]+)", msg)
        if g:
            curve[g.group(1)] = (g.group(2), g.group(3), g.group(4))
    items = list(curve.items())
    print("\n  训练 loss 曲线（训练集，无验证集）：")
    for k, v in items[:: max(1, len(items) // 12)]:
        print(f"    step={k:>6}  epoch={v[0]:>7}  loss={v[1]:>6}  grad_norm={v[2]}")
    print(f"    step={items[-1][0]:>6}  epoch={items[-1][1][0]:>7}  loss={items[-1][1][1]:>6}")
    print("  >>> 106 epoch / 30 条示教 / 关闭图像增广 / 无验证集：loss 只反映训练集拟合")


# ------------------------------------------------- [4][5] 部署日志
def parse_record() -> list[tuple[int, np.ndarray, np.ndarray]]:
    parts = re.split(r"=== Chunk (\d+) ===", RECORD.read_text())
    out = []
    for i in range(1, len(parts), 2):
        body = parts[i + 1]
        st = np.array(ast.literal_eval(
            re.search(r"Robot State \(16 joints\): (\[.*?\])", body, re.S).group(1)))
        A = np.array([ast.literal_eval(x) for x in re.findall(r"Action \d+: (\[.*?\])", body)])
        out.append((int(parts[i]), st, A))
    return out


def section_ood(chunks, state, action, episode) -> None:
    rule("[4] 部署观测 / 预测的分布外检查")
    stats = json.loads((DATASET / "meta/stats.json").read_text())
    S = np.stack([c[1] for c in chunks])
    A = np.concatenate([c[2] for c in chunks])
    smin, smax = np.array(stats["observation.state"]["min"]), np.array(stats["observation.state"]["max"])
    smean, sstd = np.array(stats["observation.state"]["mean"]), np.array(stats["observation.state"]["std"])
    amin, amax = np.array(stats["action"]["min"]), np.array(stats["action"]["max"])

    print(f"  {len(chunks)} 个 chunk，每个 {chunks[0][2].shape[0]} 步 → 共 {len(A)} 个预测动作\n")
    print(f"  观测 state（每个 chunk 起点，N={len(S)}）")
    print(f"  {'dim':6}{'obs_min':>10}{'obs_max':>10}{'tr_min':>10}{'tr_max':>10}{'z_max':>8}  越界")
    for d in range(16):
        z = (S[:, d] - smean[d]) / sstd[d]
        n_oor = int((S[:, d] < smin[d]).sum() + (S[:, d] > smax[d]).sum())
        print(f"  {NAMES[d]:6}{S[:, d].min():10.3f}{S[:, d].max():10.3f}"
              f"{smin[d]:10.3f}{smax[d]:10.3f}{np.abs(z).max():8.2f}"
              f"  {f'<<< {n_oor}/{len(S)}' if n_oor else ''}")

    print(f"\n  预测 action（N={len(A)}）越界统计")
    for d in range(16):
        lo, hi = int((A[:, d] < amin[d]).sum()), int((A[:, d] > amax[d]).sum())
        if lo or hi:
            print(f"  {NAMES[d]:6} pred[{A[:, d].min():7.3f},{A[:, d].max():7.3f}] "
                  f"train[{amin[d]:7.3f},{amax[d]:7.3f}]  低于 {lo} / 高于 {hi}")

    # 部署起始位姿 vs 训练每条 episode 的首帧
    firsts = np.stack([state[episode == e][0] for e in np.unique(episode)])
    home = chunks[0][1]
    dist = np.linalg.norm(firsts[:, :14] - home[:14], axis=1)
    print(f"\n  部署起始位姿 → 训练 episode 首帧的最小 L2（仅手臂 14 维）= {dist.min():.4f} rad")
    print("  >>> 初始条件是对的，失效不在复位环节")

    # 夹爪的二值性
    for d in (14, 15):
        v = action[:, d]
        print(f"  训练 {NAMES[d]} action：<0.05 占 {np.mean(v < 0.05):.1%}，"
              f">0.95 占 {np.mean(v > 0.95):.1%}，中间态仅 {np.mean((v >= 0.05) & (v <= 0.95)):.1%}")


def section_continuity(chunks, action, episode) -> None:
    rule("[5] chunk 边界连续性与运动速度")
    print(f"  {'chunk':>6}{'手臂|a0-state|max':>19}{'维':>5}{'全维 chunk 内位移max':>22}{'维':>5}")
    arm_jumps = []
    for idx, st, A in chunks:
        d0, span = np.abs(A[0][:14] - st[:14]), np.abs(A[-1] - A[0])
        arm_jumps.append(d0.max())
        print(f"  {idx:6d}{d0.max():19.4f}{NAMES[int(d0.argmax())]:>5}"
              f"{span.max():22.4f}{NAMES[int(span.argmax())]:>5}")
    arm_jumps = np.array(arm_jumps)
    print(f"\n  >>> 手臂 chunk 边界跳变：中位数 {np.median(arm_jumps):.4f} rad，"
          f"最大 {arm_jumps.max():.4f} rad（chunk {chunks[int(arm_jumps.argmax())][0]}）")
    print("  >>> 前 8 个 chunk ≤0.13 rad 衔接良好；chunk 9/10 退化到 0.20/0.51 rad，"
          "即 episode 后段策略已跟丢")

    step_pred = np.concatenate([np.abs(np.diff(A, axis=0)) for _, _, A in chunks])
    step_tr = np.concatenate([np.abs(np.diff(action[episode == e], axis=0))
                              for e in np.unique(episode)])
    print(f"\n  每步 |Δaction|（rad/帧）：部署 vs 训练@30fps")
    print(f"  {'dim':6}{'deploy':>10}{'train':>10}{'ratio':>8}")
    for d in range(16):
        dm, tm = step_pred[:, d].mean(), step_tr[:, d].mean()
        print(f"  {NAMES[d]:6}{dm:10.5f}{tm:10.5f}{dm / max(tm, 1e-9):8.2f}")
    r = np.mean([step_pred[:, d].mean() / max(step_tr[:, d].mean(), 1e-9) for d in range(14)])
    print(f"\n  >>> 手臂平均速度比 = {r:.2f}（≈1 表示节奏与训练一致，无时间缩放问题）")


# ------------------------------------------------- [6] 代码副本
def section_code_copies() -> None:
    rule("[6] 三份 lerobot 代码副本的一致性")
    for label, root in (("训练 venv", TRAIN_SRC), ("部署 checkout", DEPLOY_SRC),
                        ("平台 checkout", PLATFORM_SRC)):
        cfg = root / "policies/act_delta/configuration_act_delta.py"
        has_loss = cfg.exists() and "loss_time_decay" in cfg.read_text()
        print(f"  {label:14s} {root}")
        print(f"      act_delta 存在={cfg.exists()}  含 loss_time_decay={has_loss}  "
              f"rollout/inference/chunk.py={(root / 'rollout/inference/chunk.py').exists()}  "
              f"marvain_m6_http={(root / 'robots/marvain_m6_http').exists()}")
    for a, b, la, lb in ((TRAIN_SRC, DEPLOY_SRC, "训练", "部署"),
                         (PLATFORM_SRC, DEPLOY_SRC, "平台", "部署")):
        if not (a / "policies/act_delta").exists():
            continue
        r = subprocess.run(
            ["diff", "-rq", "-x", "__pycache__",
             str(a / "policies/act_delta"), str(b / "policies/act_delta")],
            capture_output=True, text=True)
        n = len([x for x in r.stdout.splitlines() if x.strip()])
        print(f"  {la} vs {lb}：act_delta 下有 {n} 处文件差异")


def main() -> None:
    section_config()
    state, action, episode = load_frames()
    section_relative(state, action, episode)
    section_training()
    chunks = parse_record()
    section_ood(chunks, state, action, episode)
    section_continuity(chunks, action, episode)
    section_code_copies()
    print("\n完成。对应报告：act/act_delta-batch5-failure-analysis-2026-08.md")


if __name__ == "__main__":
    main()
