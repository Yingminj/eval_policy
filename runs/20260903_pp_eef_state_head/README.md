# patch_policy-EEF：09-03 两个新权重 vs 08-31 旧权重

报告：[`../../../policy/experiment_report/patch_policy/patch_policy-eef-state-head-2026-09.md`](../../../policy/experiment_report/patch_policy/patch_policy-eef-state-head-2026-09.md)

回答的问题：**这两个 09-03 新权重相对旧权重变了什么，参数改动各自值多少，接下来该怎么调。**

## 评的是哪些 checkpoint

`/mnt/robot_platform/jobs/` 下四个，全部 `run/checkpoints/200000/pretrained_model`
（两个新 arm 另评 `100000`）：

| 短名 | job | head | n_obs | robot_state | 为什么在表里 |
|---|---|---|---|---|---|
| `pp_eef_state` | `patch_policy_..._505_eef_2026-08-31_13-14-33-857400` | diffusion | 2 | **on** | **基线**，09-02 报告的主角 |
| `pp_eef_nostate` | `patch_policy_..._505_eef_2026-09-03_09-44-14-723349` | diffusion | 2 | off | 新 arm 之一，**相对基线只改了一件事** |
| `pp_eef_act5` | `patch_policy_..._505_eef_2026-09-03_09-33-43-303120` | **act** | **5** | off | 新 arm 之二，**同时改了三件事** |
| `acteef_533` | `act_eef_..._533_eef_2026-08-27_14-48-42-880941` | (ACT) | 1 | — | `deploy_config_eef.yaml` 现在指着的权重 |

**参数改动出自 `config.json`，不是 `job.json`。** 三个 patch_policy 的 `job.json`
逐字段相同（200k / bs 16 / seed 1000 / lr 5.5e-5 常数），差异全在策略配置里，
65 个字段中动了 5 个。其中两个（`gpt_block_size` 2→25、`n_vqvae_training_steps`
5000→20000）**只在 `action_head="vqbet"` 下被读**，本轮两个 head 都不是 vqbet，
所以它们是死字段——`probes/config_diff.txt` §4 给了每个读取点的行号。

## 跑的命令

```bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python      # 训练这些 checkpoint 的同一解释器

$PY probes/config_diff.py                       > probes/config_diff.txt
$PY probes/deploy_stack.py                      > probes/deploy_stack.txt
$PY ../scripts_patch_policy_eval_0902/splits.py > probes/splits.txt
$PY ../scripts_patch_policy_eval_0902/ood.py    > probes/ood.txt
# check_alignment.py 对四个 checkpoint 各跑一次，见 probes/check_alignment.txt 头部

MAX_ANCHORS=50 RESULT_SUFFIX=_smoke ./run_eval.sh   # 冒烟
./run_eval.sh                                       # 全量，7 次评测，串行独占 GPU
$PY summarise.py --selftest > tables.md
```

`run_eval.sh` 串行执行，**任何时刻只有一个进程用 GPU**（09-02 那轮并行跑把
278 s 拖成 15 分钟以上，见那份 README 的"踩过的坑"）。

## harness 分叉（这轮唯一一处，顶层文件一个字节没动）

顶层 `../../offline_chunk_eval.py` **未修改**。本目录的 `offline_chunk_eval.py`
是它的副本加四处改动，完整 diff 在 `fork.diff`（72 行）。

原因只有一个：**顶层 harness 的部署重写把动作空间宽度写死成 16-D 关节。**
`send_next_action_chunk` 是按名字分的——
`arm_joint_keys = [k for k in ordered_keys if "gripper" not in k.lower()][:14]`，
在 16-D 关节里是 14 个臂 + 第 14/15 列夹爪，在 **14-D EEF 里是 12 个位姿 + 第 12/13 列夹爪**。

| # | 改了什么 | 不改会怎样 |
|---|---|---|
| 1 | 新增 `ACTION_LAYOUT` / `set_action_layout(names)`，按 deploy 自己那条 `"gripper" not in k.lower()` 规则从数据集动作名推出臂/夹爪划分 | — |
| 2 | `bridge` 从 `range(min(14, width))` 改成 `range(min(ACTION_LAYOUT["n_bridge"], width))` | 桥会改写第 12/13 列夹爪，而部署代码不碰它们（`probes/deploy_stack.txt` §4：合成 chunk 上 max\|Δ\| = 1.43） |
| 3 | `gripper_clip` 从写死的 `[:, 14:16]` 改成本动作空间真正的夹爪列 | 宽度 14 时整个 clip 是空操作，而驱动实际会 clip（max\|Δ\| = 0.50） |
| 4 | `--filter-ablation` 在宽度 < 16 时跳过 `gripper_loops` 这一级 | **直接崩**：`ValueError: Gripper indices [14, 15] are outside action width 14`。该级在机器人上本来就是 `ENABLE_REMOVE_OPEN_GRIPPER_LOOPS = False` |

**分叉在 16-D 关节空间下与顶层逐位相同**（`probes/deploy_stack.txt` 末行断言 True），
所以关节侧的历史数字不受影响；`--filters none` 路径完全没碰，
所以 `policy_raw` 与 09-02 报告可直接比——`summarise.py --selftest` 把这条写成断言。

## 两个刻度选择，以及为什么

**horizon = 60**，取自 `deploy_config_eef.yaml` 的 `inference.n_action_steps`，不是习惯值。
harness 会把它夹到策略自己的 chunk 长度：patch_policy `action_chunk_size=50` → 实际 50，
`acteef_533` `chunk_size=100` → 实际 60。**这个夹取本身是个部署事实**：
部署配置要 60 步，patch_policy 只能给 50。

**filters = `rollbacks,smoothing,bridge,gripper_clip`**，不是 `none` 也不是 `all`。
`deploy_config_eef.yaml` 用 `strategy: base` + `inference.type: chunk`，
`BaseStrategy.run`（`strategies/base.py:69`）因此走 `send_next_action_chunk`，
**重写会发生**。`strategies/core.py:52-55` 四个开关里 `gripper_loops` 和
`excursions` 是 False，`smoothing` 是无条件调用，加上驱动的夹爪 clip，
活着的就是这四级。`--filters all` 会多算两级机器人已经关掉的。

> **这否掉了 09-02 报告的一条方法学前提。** `scripts_patch_policy_eval_0902/run_eval.sh`
> 全程 `--filters none`，理由写着 EEF 路径"hands the chunk to VlaHost verbatim"。
> 代码不是这样。见 `probes/deploy_stack.txt` §1。

## 文件

| 文件 | 回答什么 | 报告章节 |
|---|---|---|
| `run_eval.sh` | 7 次评测的驱动脚本（4 个权重 @200k + 2 个 @100k + ACT 的第二个 horizon） | §4 |
| `offline_chunk_eval.py` + `fork.diff` | 部署忠实的 chunk 评测；分叉与 diff 见上 | §3.4 |
| `summarise.py` → `tables.md` | 由 `results/*.json` 出表 + 复现性断言 | 全部表 |
| `probes/config_diff.py` → `.txt` | 到底改了哪几个参数，以及哪些参数是死字段 | §2 |
| `probes/deploy_stack.py` → `.txt` | EEF 部署路径有没有 filter 栈、作用在哪几列 | §3.4 / §6 |
| `probes/splits.txt` | 评测集来源与污染（复用 `../scripts_patch_policy_eval_0902/splits.py`，未改） | §3.1 |
| `probes/ood.txt` | 越界率（复用 `../scripts_patch_policy_eval_0902/ood.py`，未改） | §3.2 |
| `probes/check_alignment.txt` | 历史索引自检，四个 checkpoint 各一次（复用同目录脚本，未改） | §3.3 |
| `results/<name>.json` / `.log` | 原始产物，一条命令一个 JSON，stdout tee 到同名 `.log` | — |

## 相对 `../scripts_patch_policy_eval_0902/` 的改动

- **harness 分叉了**（上表四处）。09-02 那轮是逐字节副本。
- **filter 设定变了**：`none` → 真实部署栈 + `--filter-ablation`。原因见上。
- **horizon 变了**：50 → 60（取自部署配置）。patch_policy 被夹回 50，所以三个
  patch_policy 的 `policy_raw` 与 09-02 的表仍然可比；`acteef_533` 额外跑了 h50 一次专门用于核对。
- **`--train-root` 多传了一个**：`505_eef` 与 `533_eef` 都传，让四次评测的 anchor 集合完全相同。
  两个训练集都不含这 53 个 episode，所以实际一个都没丢。
- `splits.py` / `ood.py` / `check_alignment.py` **原样复用**，没有拷贝到本目录。
  副作用：`splits.py` 会重写 `../scripts_patch_policy_eval_0902/splits.json`（内容相同）。

## 踩过的坑

1. **`--filter-ablation` 在 14-D 上直接崩。** `remove_open_gripper_loops` 把夹爪下标写死成
   14/15，宽度 14 时抛 `ValueError`。这不是数据问题，是那个函数只按关节空间写的。
   分叉第 4 条跳过它——而它在机器人上本来就是关的。
2. **`job.json` 看不出这次改了什么。** 三个 job 的 `job.json` 完全相同，
   差异只在 `config.json`。只看 job 记录会得出"两个权重一模一样"的错误结论。
3. **`--batch-size` 必须固定。** 扩散采样种子是**每个 batch** 重置一次，
   所以同一 anchor 的噪声取决于它在 batch 里的位置；换 batch size 数字就变。全程 8。
4. **`deploy_config_eef.yaml` 要 60 步，patch_policy 只有 50 步。** 不是 harness 截断，
   是策略本身给不出来。

## 没做的

- **没有闭环。** 全部是 teacher-forced 开环 chunk 评测。
- **没有跨空间（FK）对照。** 09-02 那轮做过，本轮四个权重全在 EEF 空间，不需要。
- **没测 50k / 150k。** 两个新 arm 只评了 100k 与 200k。
- **没测 `act` head + `use_robot_state=true`。** 这个组合至今没有被训练出来（见报告"建议"）。
- **没画图。** 表里已有 @1/@10/@25/@50 与逐级 filter 归因。
