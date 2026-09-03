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

- **预测器**（每个都在同一批 anchor、同一 padding mask 上评分）：
  `policy_raw`（策略原始输出，截断到执行窗口）、`policy_deployed`（过完部署重写）、
  空基线 `hold_state`（保持当前关节角）/ `train_mean`（训练集动作均值）
- **指标**（每个预测器都给一份）：`mae`、`rmse`、`norm_mae`（按训练集 action std 归一）、
  `mae_at_horizon[k]` / `norm_mae_at_horizon[k]`（前 k 步均值，k ∈ 1/10/25/50/horizon）、
  `mae_per_horizon[k]`（逐步曲线）、`mae_per_joint` / `norm_mae_per_joint`、
  `anchor_action_steps`（有效计数）。累加器为流式求和，anchor 数不影响显存
- **去污染**：`--train-root`（可重复）对训练集 episode 的 action 数组做 SHA1 指纹，
  命中的 eval episode 自动剔除并计数（数据集目录名不可信，只有数据本身作数）；
  `--keep-only-contaminated` 反向只留污染 episode，作同场次对照（同光照/布局，
  差异只来自记忆）
- **部署忠实**：`policy_deployed` 直接按文件路径导入机器人上的
  `lerobot/rollout/trajectory.py`（`--vlahost-src`，默认 `~/YING/lerobot_vlahost`），
  不复刻实现。滤波器固定顺序：`rollbacks` → `gripper_loops` → `smoothing` →
  `excursions` → `bridge`（K=40 三次 Hermite 桥，从实测关节角起步）→ `gripper_clip`
  （驱动层 [0,1] 钳位）。`--filters all|none|逗号列表` 选级（无论怎么写都按固定顺序生效）、
  `--filter-ablation` 逐级归因（`filt_0_clip_only` → `filt_5_bridge` 累积阶梯，
  外加 `filt_bridge_only`；共享累积结果，几乎不额外耗时）、
  `--n-action-steps` 截断到部署真正下发的执行窗口（默认 50）
- **延迟对齐**：`--latency-steps N` 把 ground truth 后移 N 拍，对应观测到首个执行点之间的
  HTTP 往返 + 推理 + vlahost 预置混合航点；0 复现原始 harness 的对齐（敏感性旋钮）
- **观测历史**：支持 `n_obs_steps > 1`（anchor 取最新帧，action 窗口偏移自动对齐；
  `patch_policy` 走 `policy.model.predict` 而非 `predict_action_chunk`）
- **采样噪声**：扩散类头部 `--seed` 固定噪声、`--seed-repeat N` 用另外 N 个种子重跑同一
  anchor 并单独计分（`seed_1..N`，仅 `patch_policy` 等采样式头部有效），给出"两个
  checkpoint 的差异是否只是采样"的下界。注意：种子按 batch 重置，anchor 的噪声取决于它在
  batch 内的位置，**跨 run 比较必须用同一个 `--batch-size`**（已记入输出 JSON）
- **轨迹导出**：`--dump-traces x.npz` 存第一个评测集的逐 anchor `pred`/`pred_raw`/`gt`/
  `state`/`valid`/`episode_index`/`frame_index`，供 `plot_traces.py` 画图；
  `--trace-anchors`（默认 200）限量，`--trace-episode N`（可重复）指定 episode
- **自检**：`--selftest` 校验累加器（padding 不泄漏、分批等价）与部署重写
  （桥从实测位姿起步、K-1 步汇合、零初速、不原地改输入、滤波器选择顺序），
  主流程前自动运行，坏 build 不出数

不能复现的部分（脚本 docstring 有详述）：闭环（每个 anchor 都从示教状态开环起步）、
图像通路（部署是 JPEG q90 + `INTER_AREA` 缩放，训练是视频帧 + `INTER_LINEAR`）、
夹爪观测（训练是指令回显，部署是标定后的真实反馈）。

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
- 常用旋钮：`--stride`（anchor 采样间隔）、`--max-anchors-per-dataset`（快速冒烟）、
  `--device`（默认 cuda）、`--latency-steps`（延迟敏感性）、`--keep-only-contaminated`（同场次对照）。
- 输出 JSON 顶层是平铺的：`aggregate`（各预测器汇总）、`per_dataset`（每个 `--dataset-root`
  一项，含 `episodes_dropped_as_contaminated` / `anchors` / 各预测器指标）、以及运行元信息
  （`checkpoint`、`policy_type`、`chunk_size`、`executed_horizon`、`latency_steps`、
  `deploy_filters`、`vlahost_src`、`seed` 等）；表格汇总与画图工具见
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

runs/ 内各目录的 README 描述了当次实验的 checkpoint、命令与关键读数。
各目录内的 harness 副本是历史快照，新实验请用顶层 `offline_chunk_eval.py`。

## 新实验流程

1. `mkdir runs/<日期>_<主题>`，写 `run.sh`（调用顶层 `offline_chunk_eval.py`）
2. 产物放 `runs/<日期>_<主题>/results/`（已被 .gitignore 忽略，只存本地）
3. 写 README.md 三段：评的是哪个 checkpoint、跑的命令、关键读数

## 用 Claude / Codex 搭实验

新实验不要手写 harness，交给 agent（Claude Code / Codex CLI）做，但**先喂读物再让它动手**。
顺序很重要：不读训练/推理代码就动手的 agent 会写出一个"看起来对"的评测——观测构造、
归一化、动作偏移三处任意一处错了，数字照样出得来，而且错得体面。

让 agent 按这个顺序读：

1. **顶层 `offline_chunk_eval.py`**（先读 docstring：它写清了能复现什么、不能复现什么），
   这是实验框架的地基，新实验是**在它之上加脚本**，不是改它、更不是重写。
2. **该 policy 的训练与推理代码**：`lerobot` 里的 `modeling_<policy>.py` /
   `configuration_<policy>.py`，重点是 `n_obs_steps`、`chunk_size` / `action_chunk_size`、
   `predict_action_chunk` 与 `select_action` 的差异（`patch_policy` 这类要走
   `policy.model.predict`）。
3. **部署链路**：`~/YING/lerobot_vlahost` 的 `lerobot/rollout/trajectory.py` 与
   `strategies/core.py`，以及对应的 `deploy_config_*.yaml`（`inference.n_action_steps`
   决定执行窗口，配错了整张表就没意义）。
4. **数据集与权重**：`meta/info.json`（fps / 特征 / episode 数）、
   `meta/stats.json`（归一化统计量，可用来验两个 checkpoint 是不是吃的同一份数据）、
   checkpoint 目录下的 `config.json` 与 `train_config.json`（训练集、步数、seed、batch size）。
5. **一份既有 run**：`runs/scripts_patch_policy_eval_0902/`（脚本 + README + 结果 JSON）
   与它对应的报告 `policy/experiment_report/patch_policy/patch_policy-eef-independent-eval-2026-09.md`，
   照这个形状产出。

几条硬约束，写进 prompt 里，agent 容易自己丢掉：

- **解释器**必须是训练该 checkpoint 的那个（一般 `/opt/robot-platform/train-venv/bin/python`）。
- **不要改顶层 harness**。要复现旧数字就把它逐字节复制进 run 目录；确实要改，
  在 run 的 README 里逐条列出改了什么、影不影响判据。
- **先冒烟再全量**：`--max-anchors-per-dataset 50` 跑通全链路，再摘掉这个旗子。
- **`--selftest` 必须过**，它在主流程前自动跑，失败就别信任何数字。
- **去污染不可省**：`--train-root` 指向该 run 真正训练用的数据集，
  目录名不算数，只有 action 指纹算数。
- **GPU 独占**：并行跑两个评测会互相抢卡，`total_seconds` 会失真。
- **数字不许编**。报告里的每个数都要能从 `runs/<dir>/results/*.json` 里指出来源；
  没跑出来的就写"没测"。

### 标准 prompt（直接复制，替换尖括号里的内容）

```text
You are setting up an offline policy evaluation experiment in the repo
/home/kewei/YING/paper/eval_policy. Work autonomously; ask me only if a
required path does not exist.

GOAL
<one sentence: the question this experiment must answer, e.g. "Does
patch_policy trained on batch_success_505_eef still beat act_eef when
scored on the independently collected 53-episode eval set?">

SUBJECTS
- checkpoints: <abs paths to .../checkpoints/<step>/pretrained_model, with a
  short name for each>
- eval dataset(s): <abs path(s)>
- training dataset(s) each checkpoint actually used: <abs path(s)>  # for --train-root
- python interpreter: <e.g. /opt/robot-platform/train-venv/bin/python>
- deploy config governing the executed window: <abs path to deploy_config_*.yaml>

STEP 1 — READ BEFORE YOU WRITE ANYTHING
Read, in this order, and summarise back to me in <=15 lines what constrains
the experiment design (do not start coding before this summary):
  1. ./offline_chunk_eval.py  — especially the module docstring: what it can
     and cannot reproduce, the deploy rewrite, the predictors and metrics.
  2. The policy's training + inference code (modeling_*.py, configuration_*.py):
     n_obs_steps, chunk size field name, predict_action_chunk vs select_action,
     whether the head samples noise.
  3. The deploy path: ~/YING/lerobot_vlahost/lerobot/rollout/trajectory.py,
     strategies/core.py, and the deploy config above (inference.n_action_steps).
  4. The datasets' meta/info.json + meta/stats.json and each checkpoint's
     config.json / train_config.json (training set, steps, seed, batch size).
  5. One prior run as the template for shape and rigour:
     runs/scripts_patch_policy_eval_0902/ (README.md, run_eval.sh, summarise.py)
     and its report at
     /home/kewei/YING/paper/policy/experiment_report/patch_policy/patch_policy-eef-independent-eval-2026-09.md

STEP 2 — BUILD THE EXPERIMENT
Create runs/<YYYYMMDD>_<topic>/ and build there. Rules:
  - Do NOT modify the top-level offline_chunk_eval.py. Call it. If you truly
    need a change, copy it into the run dir and list every diff in the run README.
  - Write a run_*.sh that drives offline_chunk_eval.py, one --out JSON per
    checkpoint/condition, stdout tee'd to a .log next to it, under results/.
  - Add small single-purpose probe scripts only where the harness cannot answer
    the question (dataset provenance, overlap/contamination, OOD rate, alignment
    self-check, cross-space comparison). One script, one question, one .txt/.json.
  - Always pass --train-root for every training set involved, and --selftest
    must pass. Set --n-action-steps from the deploy config, not from habit.
  - Use --filter-ablation when a deploy filter stack exists; use --filters none
    when the deploy path has none — and say in the README which it is and why.
  - For sampling heads, use --seed and --seed-repeat 1 so I can tell a real gap
    from sampling noise. Keep --batch-size identical across runs being compared.
  - Smoke-test everything with --max-anchors-per-dataset 50 first, then run full.
  - Run evaluations sequentially — never two on the same GPU at once.
  - Add summarise.py that turns results/*.json into the markdown tables the
    report uses. The report must contain no number that summarise.py cannot emit.

STEP 3 — RUN
Execute the smoke pass, show me the deltas you expect, then the full pass.
Report failures verbatim; never patch around a failing self-check.

STEP 4 — DELIVERABLES
  a) runs/<YYYYMMDD>_<topic>/README.md: the checkpoints evaluated, the exact
     commands, a file->purpose->report-section table, deviations from the
     previous run's harness, and any trap you hit.
  b) A report at
     /home/kewei/YING/paper/policy/experiment_report/<policy>/<policy>-<topic>-<YYYY-MM>.md,
     in Chinese, in the shape of the template report above:
       - header block: eval set (with ep/frame/anchor counts), checkpoints table,
         measurement date + machine + interpreter, path to scripts+raw results,
         which prior report this supersedes or corrects
       - "1. 结论": <=5 numbered claims, each one carrying its own numbers
       - a section auditing the eval set (provenance, contamination, OOD rate,
         alignment) before any accuracy table
       - the main tables (@1/@10/@25/@50, rmse, vs null), grouped-error tables
         where the action dims have mixed units, and the deploy-window number
       - a mechanism section explaining *why*, not just *what*
       - "建议": what to deploy, what to stop investing in, what is still untested
     Every number traceable to results/*.json. State null baselines alongside every
     accuracy number. If a prior report is contradicted, say so explicitly and
     name the section. Write "not measured" rather than guessing.

Finish by printing the report path and the three headline numbers.
```
