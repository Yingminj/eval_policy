# scripts_patch_policy_eval_0902

`../../patch_policy/patch_policy-eef-independent-eval-2026-09.md` 的测量脚本与原始结果。

这一轮回答 08-31 报告 §3.2 留下的窟窿：**EEF 权重从来没有在独立场次上被评过**。
两个前提条件在 09-02 齐了：

| 变化 | 东西 |
|---|---|
| 独立评测集有 EEF 版了 | `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data_eef` |
| ACT-EEF 基线换成同训练集 | `/mnt/robot_platform/jobs/act_eef_tidy_up_stationery_le_batch_success_505_eef`（200 k，seed 1000，bs 16） |

于是 08-31 那张"公平对照"表的两个混淆项（随机交错划分 / 基线少 39% 数据）都能被拆掉。

| 文件 | 作用 | 报告章节 |
|---|---|---|
| `splits.py` → `splits.txt`, `splits.json` | 新 EEF 评测集与关节评测集是不是同一批录制；与全部训练集的交集 | §3.1 |
| `fk_provenance.py` → `fk_provenance.txt` | EEF 数据集是不是关节数据集的 FK 重投影（决定两个动作空间可不可比） | §3.2 |
| `ood.py` → `ood.txt` | 独立评测集 vs 08-31 随机留出集，各有多少帧落在训练集 MIN_MAX 盒子外 | §3.3 |
| `check_alignment.py` → `check_alignment.txt` | 历史索引自检，两个 checkpoint 都过 | §3.4 |
| `offline_chunk_eval.py` | 部署忠实的 chunk 评测，`../scripts_patch_policy_eval_0831/` 的**逐字节副本** | §3.4 |
| `run_eval.sh` → `pp_*`, `acteef_*` | 主精度测量（含 `deploy_config_eef.yaml` 指着的 `acteef_533`） | §4 |
| `cross_space.py` / `run_cross.sh` → `cross_*.json` | 把关节 chunk 过一遍同一份 FK，两个动作空间在**同一个物理量**上比 | §5 |
| `run_anchor.sh` / `anchor.py` → `anchor_*.json` | chunk 首帧与实测位姿的偏差，对任意策略类型都能跑（`probe_conditioning.py` 只吃 patch_policy） | §6 |
| `summarise.py` → `tables.md` | 由 JSON 生成报告里的表 | — |

解释器必须是 `/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的同一环境）。

```bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
$PY splits.py > splits.txt
$PY fk_provenance.py > fk_provenance.txt
$PY ood.py > ood.txt
./run_eval.sh          # 主表，约 8 分钟（扩散头两个 seed 占了大半）
./run_cross.sh         # 跨空间表，约 12 分钟
./run_anchor.sh        # 位姿锚定，约 12 分钟
$PY summarise.py all > tables.md
```

## 相对 `../scripts_patch_policy_eval_0831/` 的改动

- `offline_chunk_eval.py` **未改**，所以本轮 EEF 数字与 08-31 的表直接可比（同一 harness、同一 stride、同一 filter 设定）。
- `check_alignment.py` 两处：`n_obs_steps == 1` 时 `observation.state` 没有时间轴（`act_eef` 就是这种），补一维；
  `action_chunk_size` 是 patch_policy 的字段名，ACT 叫 `chunk_size`，取其一。
  两处都只影响**能不能跑**，不改判据。
- 新增 `fk_provenance.py`、`cross_space.py`、`anchor.py`。

## 一次踩过的坑

同一台机器上并行跑两份 `run_eval.sh` 会互相抢 GPU，第一个 run 从 278 s 涨到 15 分钟以上还没完。
表里的 `total_seconds` 是独占 GPU 的重跑值。
