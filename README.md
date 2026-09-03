# eval_policy

policy 离线评测仓库。顶层是唯一维护的评测脚本，`runs/` 是历次实验（脚本 + 原始产物），只读存档。

## 目录结构

```
offline_chunk_eval.py   # 核心：离线开环 chunk 精度评测（policy 无关）
runs/                   # 历次实验存档，每个目录对应一份实验报告
```

## 核心脚本：offline_chunk_eval.py

源自 `runs/scripts_patch_policy_eval_0831/`（四代 harness 中功能最全的版本）。
对数据集中按 stride 抽样的 anchor 帧，复现训练时的观测构造与归一化，开环预测整段
action chunk，与示教动作对比。支持 `act` / `act_dit` / `patch_policy` / `vita`。

主要能力：

- **指标**：`mae`、`mae_at_horizon[k]`（前 k 步累计）、`mae_per_horizon[k]`、
  `mae_per_joint`、`norm_mae`；基线 `hold_state` / `train_mean`
- **去污染**：`--train-root`（可重复）对训练集 episode 做 action 指纹，
  污染 episode 自动剔除并计数
- **部署忠实**：`policy_deployed` 预测器复现部署侧轨迹滤波器（gripper clip、
  rollback/loop 移除、平滑、K 步 Hermite 桥），`--filters` 选级、
  `--filter-ablation` 逐级归因、`--n-action-steps` 截断执行窗口
- **观测历史**：支持 `n_obs_steps > 1`（anchor 取最新帧，action 偏移自动对齐）
- **采样噪声**：扩散类头部 `--seed` / `--seed-repeat`
- **自检**：`--selftest` 校验累加器与部署重写逻辑，主流程前自动运行

### 用法

```bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python    # 训练 checkpoint 的解释器
D=/mnt/robot_platform/datasets/tidy_up_stationery_le

$PY offline_chunk_eval.py --checkpoint /mnt/robot_platform/jobs/<job>/run/checkpoints/last/pretrained_model \
  --train-root $D/<训练集> \
  --dataset-root $D/<评测集> \
  --stride 20 --batch-size 32 --num-workers 8 \
  --n-action-steps 50 --filter-ablation \
  --out results/<名称>.json
```

注意：

- 解释器必须是训练该 checkpoint 的环境；`act_dit`（EMA）需
  `export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src` 并用 `conda run -n lerobot`。
- `--train-root` 指向该 run 实际训练用的数据集，不同 run 不同。
- 输出 JSON 含 `aggregate` / `per_dataset` / `config`；表格汇总与画图工具见
  `runs/scripts_patch_policy_eval_0831/summarise.py`、`runs/scripts_act_eval_test/plot_horizon.py`。

## runs/ 存档索引

| 目录 | 内容 | 对应报告（policy 仓库 experiment_report/） |
|---|---|---|
| `scripts_act_eval_test` | 第一代 harness + 绘图工具，ACT baseline 评测 | act/ |
| `scripts_act_eval_test_fix` | 第二代：部署忠实版（滤波器/Hermite 桥） | act/ |
| `scripts_act_dit_eval_fix` | ACT-DiT 部署忠实评测 runner（未分叉 harness） | act_dit/ |
| `scripts_act_dit_probe` | encoder 塌缩诊断四件套（encoder/conditioning/ablation/sampling） | act_dit/ |
| `scripts_act_dit_lowlr` | 低学习率复训的原始数据（无脚本） | act_dit/ |
| `scripts_act_delta_audit` | act_delta 部署失效审计（batch5 / rel100） | act/ |
| `scripts_act_layer_bench` | ACT 架构基准（层数/backbone/FLOPs/梯度/位置嵌入） | — |
| `scripts_deploy_audit` | 部署链路审计（轨迹滤波器独立副本、消融、延迟） | — |
| `scripts_fail_data` | 失败数据分析（batch_fail_72 vs batch_success_361） | — |
| `scripts_patch_policy_probe` | patch_policy 早期诊断四件套 | patch_policy/ |
| `scripts_patch_policy_compare` | ACT 头 vs 扩散头对比（消融/扫描/延迟） | patch_policy/ |
| `scripts_patch_policy_eval_fix` | 第三代 harness（+观测历史） | patch_policy/ |
| `scripts_patch_policy_eval_0831` | 第四代 harness（顶层脚本的来源） | patch_policy/ |
| `scripts_patch_policy_eval_0902` | 与 0831 逐字节相同，EEF 独立评测集 | patch_policy/ |
| `scripts_vita_chunk` | VITA chunk 部署平滑性分析（seam jump 等指标族） | vita/ |

runs/ 内各目录的 README 描述了当次实验的 checkpoint、命令与关键读数。
各目录内的 harness 副本是历史快照，新实验请用顶层 `offline_chunk_eval.py`。

## 新实验流程

1. `mkdir runs/<日期>_<主题>`，写 `run.sh`（调用顶层 `offline_chunk_eval.py`）
2. 产物放 `runs/<日期>_<主题>/results/`（已被 .gitignore 忽略，只存本地）
3. 写 README.md 三段：评的是哪个 checkpoint、跑的命令、关键读数
