# ACT-DiT 观测编码器塌缩 — 测量脚本

报告：[`../../act_dit/act_dit-encoder-collapse-2026-08.md`](../../act_dit/act_dit-encoder-collapse-2026-08.md)

四个脚本，回答四个不同的问题。都带 `--selftest`，主流程运行前会自动跑一遍。

| 脚本 | 问题 | 代价 |
|---|---|---|
| `probe_encoder_collapse.py` | encoder 还活着吗？（逐层增益、输出幅值、图像敏感度） | 秒级，只读权重 + 一次前向 |
| `probe_conditioning.py` | 发出的 chunk 到底来自哪条通路？（图像 / encoder token / adaLN） | ~1 min，需要数据集 |
| `train_ablation.py` | 是什么把 encoder 关掉的？（受控短训练，逐 arm） | ~40 min / arm @ 8000 步 |
| `sweep_sampling.py` | 不重训能改多少？（积分步数、采样平均） | ~20 min，取决于 grid |

## 环境

`act_dit` 只存在于 `lerobot_vlahost` 和 `robot_data_platform/lerobot` 两份代码里，
`lerobot` conda env 自带的 lerobot 没有它。所以都要：

```bash
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src
conda run -n lerobot python <script> ...
```

`sweep_sampling.py` 另外要把 `../scripts_act_eval_test` 也加进 `PYTHONPATH`
（它复用 `offline_chunk_eval.py` 的累加器、去污染过滤和 `hold_state` 基线）。

mgmt01 的 `/opt/robot-platform/train-venv` **不能用**：那份 `act_dit` 落后于 repo，
没有 EMA 代码，加载这个 checkpoint 会直接失败。

## 关键读数

- `probe_encoder_collapse.py` 的 **`enc3 img-sens`**：健康的 ACT 是 ~5e-2，塌掉的
  ACT-DiT 是 ~2e-6。这一个数就够判死刑。
- `probe_conditioning.py` 的 **`images_swapped`**：占帧间自然差异的比例。必须固定初始
  噪声才有意义——不固定的话采样方差会盖过一切，把 0 读成 9%。
- `train_ablation.py` 的 **`image_sensitivity` 随 step 的走向**：500 步内就能分出胜负，
  不用等 loss 收敛。
