# scripts_patch_policy_eval_0831

`../../patch_policy/patch_policy-vqbet-and-eef-2026-08.md` 的测量脚本与原始结果。

评的是 2026-08-31 更新的两个 patch_policy 权重：

| 名字 | job | 动作空间 | 头 | 步数 |
|---|---|---|---|---|
| `vqbet_*` | `patch_policy_..._batch_success_361_2026-08-30_11-19-40-151329` | 16-D 关节 | `vqbet` | **100 k / 200 k（仍在训练）** |
| `eef_*` | `patch_policy_..._batch_success_505_eef_2026-08-31_13-14-33-857400` | 14-D 末端位姿 | `diffusion` | 200 k（已完成） |

**两块数字不可互相比较**：动作空间不同（弧度 vs 米+弧度+[0,1]），
评测集也不同（见下），只有各自与自己 null 基线的比值是跨空间可读的。

| 文件 | 作用 | 报告章节 |
|---|---|---|
| `splits.py` → `splits.txt`, `splits.json` | 用 `action` 与 `observation.velocity` 指纹判定每个 checkpoint 能合法评测的 episode | §3.2 |
| `ood.py` → `ood.txt` | EEF 留出集有多少帧落在训练集 MIN_MAX 盒子之外 | §3.3 |
| `offline_chunk_eval.py` | 部署忠实的 chunk 评测，`../scripts_patch_policy_eval_fix/` 的副本 + `--train-root` 可重复 | §3.1 |
| `check_alignment.py` | 历史索引自检（观测锚点 = 最新帧，动作偏移 = `n_obs_steps - 1`） | §3.1 |
| `run_eval.sh` → `*.json`, `*.log` | 主精度测量（留出集、公平对照） | §4, §5 |
| `run_eval2.sh` → `*_intrain.json` | 训练集内对照，用来把记忆和泛化分开 | §5.4 |
| `held.py` → `held.txt` | 相邻帧命令逐位相同的比例（`check_alignment.py` 的负控制为什么会误报） | §3.1 |
| `vq_floor.py` → `vq_floor.json` | 把真值 chunk 送进冻结码本，量 `vqbet` 的表达上限 | §8 |
| `run_extras.sh` | `vq_floor.py` + `latency.py`，按顺序跑（延迟要独占 GPU） | §7, §8 |
| `probe_conditioning.py` / `run_probe.sh` → `probe_*.json` | 条件化探针，含本体感觉干预 | §6 |
| `latency.py` → `latency.json` | batch-1 推理延迟（需独占 GPU） | §7 |
| `summarise.py` → `tables.md` | 由 JSON 生成报告里的表 | — |

解释器必须是 `/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的同一环境）。

```bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
/opt/robot-platform/train-venv/bin/python splits.py > splits.txt
/opt/robot-platform/train-venv/bin/python ood.py > ood.txt
./run_eval.sh; ./run_eval2.sh; ./run_probe.sh
./run_extras.sh          # vq_floor + latency，latency 要求 GPU 空闲
/usr/bin/python3 summarise.py all > tables.md
```

## 相对 `../scripts_patch_policy_eval_fix/offline_chunk_eval.py` 的改动

两处。

1. `--train-root` 改成可重复（`action="append"`），排除集是各个 root 的并集。
   EEF 的公平对照需要同时排掉两个训练集（`batch_success_505_eef` 和 `batch_success_361_eef`），
   单个 root 做不到。
2. `check_alignment.py` 的负控制（"`action[t-1]` 必定不等于 `action[t]`"）从"每个探针都必须成立"
   放宽成"8 个探针里至少 1 个成立"。EEF 录制里 2.7% 的相邻帧命令逐位相同（`held.py`，
   关节集 2.55–2.80%，量级相同），原断言会在这种帧上误报。

其余逐字节相同，所以 `vqbet_*` 的数字与 08-30 报告的表直接可比。

## 评测集的选取

- **关节侧**：`batch_success_53_eval_data`（53 episodes / 40 132 帧），与 08-30 报告同一份，
  同一 stride、同一 2007 个 anchor。
- **EEF 侧**：没有 EEF 版的 53 集——它从未从 rosbag 转过，原始 bag 也不在本集群上
  （`meta/conversion_manifest.json` 指向部署机 `snorlax` 的路径）。
  退而求其次用 `batch_success_533_eef` 里 **action 指纹不在 `batch_success_505_eef` 中的 67 个 episode**。
  这是一个**随机划分**（留出的 episode 序号散布在 0–506 之间，与训练 episode 同场次交错），
  比 53 集那种独立场次**容易**，所以 EEF 的绝对数字不能和关节侧的绝对数字并排读。
