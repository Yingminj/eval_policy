# acc@τ 与末端位姿误差：两个新指标，四个旧权重

报告：[`../../../policy/experiment_report/patch_policy/patch_policy-acc-tau-eef-pose-metrics-2026-09.md`](../../../policy/experiment_report/patch_policy/patch_policy-acc-tau-eef-pose-metrics-2026-09.md)

回答的问题：**给 offline chunk 评测加两个 MAE 之外的指标，它们说了什么 MAE 没说的话；
以及加完之后，历史数字有没有动。**

评的 checkpoint 与 09-03 那轮**完全相同**（见
[`../20260903_pp_eef_state_head/README.md`](../20260903_pp_eef_state_head/README.md)），
本轮不引入任何新权重，只换尺子。

## 两个新指标

| 指标 | 开关 | 为什么 |
|---|---|---|
| `acc@τ` | **永远算** | 每 batch 一次广播比较，比"要不要算"的分支还便宜；见报告 §2.1 |
| 末端位姿误差 | **按适用性自动开** | 要一遍 FK 或一次位姿切分；动作空间不支持就打印原因跳过；见报告 §2.2 |

`resolve_eef()` 按动作维度名字分派三条路：14-D EEF → 直接切分（**本轮走的这条**）、
16/54-D 关节 → MJCF 正运动学、其余 → 跳过。

## harness：改的是顶层，不是分叉

与 09-03 那轮**相反**：那轮分叉（`fork.diff` 72 行）并保持顶层
`offline_chunk_eval.py` 一个字节没动；本轮直接改顶层，
diff 见 [`harness.diff`](harness.diff)（+459 / −7）。

允许这么做的唯一理由是**十个已发表标量逐位复现**（报告 §3）。
新增的全是新键，旧键的算法一行没动。

**没有修的：** 顶层部署重写仍把动作空间写死成 16-D 关节（09-03 报告 §3.4 的四条）。
所以本轮全程 `--filters none`，**没有部署列**。这是下一轮最该做的事。

## 跑的命令

```bash
./run_eval.sh          # 五条主 run（四个权重，acteef_533 另评 h50）
probes/seed_noise.sh   # 两个扩散权重各重抽一次，给新指标定判读线
```

主 run 与 09-03 的 `run_eval.sh` 唯一的差别是 `--filters none`（那轮是
`rollbacks,smoothing,bridge,gripper_clip --filter-ablation`）。
`--stride 20 --batch-size 8 --seed 0` 与两个 `--train-root` 全部保持一致，
**anchor 集合与 09-03 逐个相同**（2007 个，丢弃 0）。

`probes/seed_noise.sh` 用两个独立 `--seed` 进程取代 `--seed-repeat`：
结果与 09-03 §4.5 逐位一致（0.03721→0.03757、0.03916→0.03854），
而且**必须串行跑**——并发跑会几个进程抢同一张卡，看起来像挂了实际只是慢。

## 结果

`results/*.json` 按仓库 `.gitignore` 约定只留在本地。
`summarise.py` 从这些 JSON 重建报告里的四张表：

```bash
python summarise.py
```

| 文件 | 是什么 |
|---|---|
| `results/pp_eef_state.json` 等五个 | 主表，`--seed 0` |
| `results/*_seed1.json` 两个 | 判读线用的第二次抽样 |
| `harness.diff` | 顶层 harness 的全部改动 |

## 自检

```bash
python ../../offline_chunk_eval.py --selftest
```

四组：`accumulator`、`eef slice`（π/−π 绕回、欧氏距离、左右与 reshape 不串台）、
`eef pose error`（FK 链左右隔离、单铰链转角）、`deploy rewrite + filter selection`。
