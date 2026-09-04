# ACT-DiT — flow matching + lr 1e-5 + EMA，以及 state 出 adaLN 的第二个 arm

报告：[`../../../policy/experiment_report/act_dit/act_dit-fm-lowlr-2026-09.md`](../../../policy/experiment_report/act_dit/act_dit-fm-lowlr-2026-09.md)

## 评的是哪些 checkpoint

`/mnt/robot_platform/jobs/` 下五个，全部取 `run/checkpoints/200000/pretrained_model`：

| 短名 | job | 为什么在这张表里 |
|---|---|---|
| `fm_lowlr` | `act_dit_..._2026-08-31_05-22-33-507248` | **本次主角**：lowlr §6.1 点名要跑的 (fm, 1e-5, EMA)，训完从没评过 |
| `fm_lowlr_noadaln_rk4` | `act_dit_..._2026-08-31_06-17-39-059317` | 第二个 arm：`state_in_adaln=false` **且** 积分器 euler→rk4（两变量） |
| `fm_hilr` | `act_dit_..._2026-08-27_04-32-02-437338` | 已发表参照，flowmatching-deployed 的主角 |
| `diff_lowlr` | `act_dit_..._2026-08-24_06-21-05-197422` | 已发表参照，lowlr 的主角 |
| `act_baseline` | `act_..._2026-08-17_12-42-42-097328` | 验收参照 |

**参照 arm 是重跑的，不是从旧报告抄的。** 旧数字出自
`runs/scripts_act_eval_test/` 的第一代 harness；跨代拼一张表就不是同一把尺子。
重跑复现了已发表的值（见下"一致性核对"），所以两边可以互相担保。

## `state_in_adaln` 的坑

七个 act_dit 权重里有六个的 `config.json` 早于这个字段，按
`configuration_act_dit.py:56` 的说明，**缺字段 = 训练时 state 走了 adaLN**，
直接加载会在 adaLN `Linear` 上报 shape 错误。

`make_ckpt_shims.py` 为每个权重建一个影子目录（`ckpt/<name>/`）：除 `config.json`
外全是指向归档权重的符号链接，`config.json` 是补了 `state_in_adaln` 的副本。
**归档权重一个字节都没动。**

判据不取自 config，取自权重本身：adaLN 输入宽度 256 = 只有 timestep（False），
256 + `dim_model` = state 也在（True）。脚本断言两者一致，`fm_lowlr_noadaln_rk4`
自己声明的 `false` 与它 256 的宽度互为交叉验证。

## 跑的命令

```bash
./run.sh                                    # 10 次评测：5 个 checkpoint × horizon {100, 50}
/opt/robot-platform/train-venv/bin/python summarise.py --selftest
```

`run.sh` 调用顶层 `../../offline_chunk_eval.py`，**未分叉、未修改**。
两个 horizon 是因为两篇前作用了两把尺子：`h100` 对应 lowlr §4（全 chunk），
`h50` 对应 flowmatching-deployed §3（执行窗口 + 部署重写）。
`--train-root` 指纹去污染：53 个 episode 实测 **0 个**被丢。

探针（`../scripts_act_dit_probe/`，同样未分叉）：

```bash
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
for n in fm_lowlr fm_lowlr_noadaln_rk4 fm_hilr; do
  $PY ../scripts_act_dit_probe/probe_encoder_collapse.py --checkpoint ckpt/$n --out results/enc_$n.json
done
$PY ../scripts_act_dit_probe/probe_conditioning.py --checkpoint ckpt/fm_lowlr \
  --dataset-root $D/batch_success_361 --n-anchors 32 --out results/cond_fm_lowlr.json
```

> `probe_conditioning.py` **跑不了 `fm_lowlr_noadaln_rk4`**：它按 road 3 存在来构造
> 768 维条件向量，而这个权重根本没有 road 3（`RuntimeError: mat1 and mat2 shapes
> cannot be multiplied (32x768 and 256x3072)`）。这不是 bug，是那个 arm 的定义。
> 没有为此分叉探针——`probe_encoder_collapse.py` 的 img-sensitivity 已经把
> "拿掉 adaLN 之后编码器是不是更看图了"回答掉了（答案是没有，见报告 §4）。

## 关键读数

评测集 `batch_success_53_eval_data`，53 ep / 2007 anchor / stride 20 / 0 污染。

**horizon 100，`policy_raw`**（MAE，rad，越小越好）

| run | mae | @1 | @10 | @50 | vs null | 穿过 null |
|---|---:|---:|---:|---:|---:|---:|
| **`fm_lowlr`** | **0.06730** | **0.01611** | **0.02352** | **0.04727** | **2.48×** | **第 2 帧** |
| `act_baseline` | 0.06848 | 0.02529 | 0.03115 | 0.05112 | 2.44× | 第 6 帧 |
| `fm_lowlr_noadaln_rk4` | 0.07073 | 0.02625 | 0.03139 | 0.05244 | 2.36× | 第 6 帧 |
| `fm_hilr` | 0.07373 | 0.04707 | 0.04719 | 0.05596 | 2.27× | 第 11 帧 |
| `diff_lowlr` | 0.10546 | 0.06895 | 0.07705 | 0.09415 | 1.59× | 第 25 帧 |
| `hold_state` | 0.16721 | 0.01556 | 0.03174 | 0.09705 | 1.00× | — |

**horizon 50，raw → deployed**

| run | raw | deployed | 桥的作用 | vs null |
|---|---:|---:|---:|---:|
| **`fm_lowlr`** | **0.04734** | 0.04961 | **+4.8%（有害）** | 1.96× |
| `act_baseline` | 0.05112 | 0.05088 | −0.5% | 1.91× |
| `fm_lowlr_noadaln_rk4` | 0.05237 | 0.05160 | −1.5% | 1.88× |
| `fm_hilr` | 0.05630 | 0.05240 | −6.9% | 1.85× |
| `diff_lowlr` | 0.09418 | 0.07729 | −17.9% | 1.26× |

**编码器体检**（`probe_encoder_collapse.py`，enc3 = 整条观测通路的总开关）

| run | mean\|γ\| | signal | img-sensitivity |
|---|---:|---:|---:|
| `fm_lowlr` | 0.9670 | 0.76084 | **0.158464** |
| `act_baseline` | 1.0037 | 0.81420 | 0.049903 |
| `fm_lowlr_noadaln_rk4` | 0.9678 | 0.70876 | 0.057519 |
| `fm_hilr` | 0.1321 | 0.00382 | **0.000001**（塌） |

**通路归因**（`probe_conditioning.py`，`delta_frac_of_spread`）

| run | images | state→adaLN | state→token |
|---|---:|---:|---:|
| `fm_lowlr` | 0.3083 | 0.3581 | 0.0053 |
| `diff_lowlr`（存档 `../scripts_act_dit_lowlr/cond_lowlr_200000.json`） | 0.2706 | 0.3662 | 0.0281 |
| `fm_hilr` | **0.0000** | 0.5606 | 0.0000 |

## 一致性核对

同一把尺子重跑已发表的 arm，四个数都对上了：

| 量 | 本次 | 已发表 | 出处 |
|---|---:|---:|---|
| `act_baseline` h100 raw | 0.06848 | 0.0685 | lowlr §4 |
| `act_baseline` h50 raw | 0.05112 | 0.0511 | flowmatching-deployed §1 |
| `diff_lowlr` h100 raw | 0.10546 | 0.1056 | lowlr §4 |
| `diff_lowlr` 通路归因 images / adaLN | 0.271 / 0.366 | 27.1% / 36.6% | encoder-collapse §8 P1 注 |

`summarise.py --selftest` 把第一条写成断言（容差 5e-4），harness 换代就会红。

**重复性**：`fm_lowlr` h100 原地重跑一次，mae 0.06730 → 0.06733（**0.04%**），
mae@1 0.016109 → 0.016118（0.06%）。报告里任何小于 0.5% 的差都不作数。
h50 与 h100 的 mae@1 相差 0.6%（0.01620 vs 0.01611），同属这个量级，不解读。

## 文件

| 文件 | 内容 |
|---|---|
| `run.sh` | 10 次评测的驱动脚本 |
| `make_ckpt_shims.py` | `state_in_adaln` 影子 checkpoint；含 `_demo()` 自检 |
| `summarise.py` | 出表 + 复现性断言 |
| `ckpt/<name>/` | 影子目录（符号链接 + 改过的 `config.json`），非权重 |
| `results/eval53_<name>_h{100,50}.json` | 动作误差原始产物 |
| `results/enc_<name>.json` | 编码器体检 |
| `results/cond_<name>.json` | 通路归因 |

## 没做的

- **没画图**。前作的 `eval53_horizon_*.png` 这次没出，表里已有 @1/@10/@25/@50
  与穿过 null 的帧号；要曲线用 `../scripts_act_eval_test/plot_horizon.py`。
- **没测 50k/100k/150k**。两个 08-31 job 磁盘上有中间 checkpoint，本次只评 200k。
- **没跑 `sweep_sampling.py`**。前作已定论"调大推理步数是负收益"，本次没有理由重开。
- **没有闭环**。全部是 teacher-forced 开环 chunk 评测，顶层 harness 的 docstring 有详述。
