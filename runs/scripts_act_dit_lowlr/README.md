# ACT-DiT 低学习率 + diffusion 目标复训 — 原始测量数据

报告：[`../../act_dit/act_dit-lowlr-diffusion-2026-08.md`](../../act_dit/act_dit-lowlr-diffusion-2026-08.md)

这里只有数据，**没有脚本**。四个脚本都在别处，且都是共用工具，不在此处分叉：

| 脚本 | 位置 |
|---|---|
| `probe_encoder_collapse.py` / `probe_conditioning.py` / `sweep_sampling.py` | [`../scripts_act_dit_probe/`](../scripts_act_dit_probe/) |
| `offline_chunk_eval.py` / `plot_horizon.py` | [`../scripts_act_eval_test/`](../scripts_act_eval_test/) |

`probe_conditioning.py` 和 `sweep_sampling.py` 在本次测量中被就地改成按 `cfg.objective`
分派（flow matching 走 Euler，diffusion 走 DDIM），细节见报告 §7。

## 文件

**评测集口径**：`eval53_*` 是现行结果，评测集为
`tidy_up_stationery_le/batch_success_53_eval_data`（53 ep，08-21 录制，与训练集 0 重叠）。
不带前缀的 `lowlr_*` / `hilr_*` 是 08-27 首测，held-out 用的是 `batch_1`–`batch_4`，
**已被取代**，保留是因为报告 §4.3 拿两者的落差做了一条结论。两套数字不要混用。

| 文件 | 产出脚本 | 内容 |
|---|---|---|
| `eval53_lowlr_{050000,100000,150000,200000}.json` | `offline_chunk_eval.py` | **现行**：本次 4 个 checkpoint 在评测集上的动作误差 |
| `eval53_hilr_200000.json` | `offline_chunk_eval.py` | **现行**：08-22（diffusion, lr 1e-4）对照 |
| `eval53_flowmatching_200000.json` | `offline_chunk_eval.py` | **现行**：08-20（flow matching）对照 |
| `eval53_sweep_lowlr.json` | `sweep_sampling.py` | **现行**：DDIM 推理步数 × 采样平均 |
| `eval53_horizon_lowlr.png` | `plot_horizon.py` | **现行**：四条 horizon 曲线 + `hold_state` |
| `../scripts_act_eval_test/eval53_act_baseline.json` | `offline_chunk_eval.py` | **现行**：ACT baseline 对照 |
| `enc_lowlr.json` | `probe_encoder_collapse.py` | 7 个 checkpoint 的逐层 encoder 增益 / 幅值 / 图像敏感度 |
| `cond_lowlr_{050000,100000,150000,200000}.json` | `probe_conditioning.py` | 本次 4 个 checkpoint 的通路归因 |
| `cond_hilr_200000.json` | `probe_conditioning.py` | 08-22（diffusion, lr 1e-4）对照 |
| `lowlr_heldout.json` / `lowlr_intrain.json` | `offline_chunk_eval.py` | 旧 held-out 263 ep（已取代）/ 训练集内（仍在用） |
| `lowlr_heldout_{050000,100000,150000}.json` | `offline_chunk_eval.py` | held-out MAE 随训练步数的走向（报告 §4.1） |
| `hilr_heldout.json` / `hilr_intrain.json` | `offline_chunk_eval.py` | 08-22 对照，本次新测——分离 LR 与 objective 的唯一依据 |
| `sweep_lowlr.json` | `sweep_sampling.py` | DDIM 推理步数 × 采样平均（10/50 × 1/8） |
| `horizon_lowlr.png` | `plot_horizon.py` | 五条 held-out horizon 曲线 |

## 复现

```bash
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src
CK=/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-24_06-21-05-197422/run/checkpoints/200000/pretrained_model
D=/mnt/robot_platform/datasets/tidy_up_stationery_le

# 编码器体检（秒级）
conda run -n lerobot python ../scripts_act_dit_probe/probe_encoder_collapse.py --checkpoint $CK

# 通路归因（~1 min）
conda run -n lerobot python ../scripts_act_dit_probe/probe_conditioning.py \
  --checkpoint $CK --dataset-root $D/batch_success_361 --n-anchors 32

# 评测集动作误差（~20 s），--train-root 触发指纹去污染（这个集实测 0 个被丢）
conda run -n lerobot python ../scripts_act_eval_test/offline_chunk_eval.py --checkpoint $CK \
  --dataset-root $D/batch_success_53_eval_data \
  --train-root $D/batch_success_361 --stride 20 --out eval53_lowlr_200000.json

# 推理旋钮扫描（~15 min）
PYTHONPATH=$PYTHONPATH:../scripts_act_eval_test conda run -n lerobot python \
  ../scripts_act_dit_probe/sweep_sampling.py --checkpoint $CK \
  --dataset-root $D/batch_success_53_eval_data --train-root $D/batch_success_361 \
  --steps 10 --steps 50 --samples 1 --samples 8 --stride 100
```

`/opt/robot-platform/train-venv`（mgmt01 那份）**不能用**：没有 EMA 代码，加载会直接失败。

## 关键读数

- `enc_lowlr.json` 的 **enc3 `image_sensitivity`**：健康 ≥1e-2（本次 2.2e-2，ACT 5.0e-2），
  塌缩 ~1e-6。一个数定生死。
- `cond_*.json` 的 **`delta_frac_of_spread.images_swapped`**：健康 >20%（本次 27.1%），
  塌缩 0%。必须固定初始噪声才有意义。
- `eval53_*.json` 的 **`aggregate.policy.mae_at_horizon["10"]` 对比 `hold_state` 的同一项**：
  前 10 帧是真正会被执行的那一段。本次是 0.0771 vs 0.0317——全 horizon 赢 1.58×，
  但短 horizon 仍然不如原地不动，**只看 `mae` 会读反**。
