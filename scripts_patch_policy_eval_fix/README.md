# scripts_patch_policy_eval_fix

`../../patch_policy/patch_policy-state-and-window-2026-08.md` 的测量脚本与原始结果。

评的是 2026-08-30 的两个新 patch_policy 权重（`new_state5` / `new_obs2`），
以及它们的三个参照（`prev_diffusion` / `prev_act_head` / `act_baseline`），
评测集 `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data`。

| 文件 | 作用 | 报告章节 |
|---|---|---|
| `offline_chunk_eval.py` | 部署忠实的 chunk 评测，`../scripts_act_eval_test_fix/` 的 fork + 观测历史支持 | §3.1 |
| `check_alignment.py` | 历史索引自检（观测锚点 = 最新帧，动作偏移 = `n_obs_steps - 1`） | §3.1 |
| `run_eval.sh` → `*.json`, `*.log`, `tables_h50.md` | horizon 50 的精度、滤波阶梯、采样噪声 | §4, §6 |
| `run_eval_h8.sh` → `*_h8.json`, `tables_h8.md` | horizon 8 = `deploy_config_patch_policy.yaml` 的真实执行窗口 | §5, §6 |
| `probe_conditioning.py` / `run_probe.sh` → `probe_*.json` | 模型条件化探针，含本体感觉干预 | §7 |
| `latency.py` → `latency.json` | batch-1 推理延迟（需独占 GPU） | §8 |
| `summarise.py` | 由 JSON 生成报告里的表 | — |

解释器必须是 `/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的同一环境）。

```bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
/opt/robot-platform/train-venv/bin/python check_alignment.py
./run_eval.sh; ./run_eval_h8.sh; ./run_probe.sh
/opt/robot-platform/train-venv/bin/python latency.py --out latency.json   # GPU 必须空闲
/usr/bin/python3 summarise.py .        # horizon 50
/usr/bin/python3 summarise.py . _h8    # horizon 8
```

## 相对 `../scripts_act_eval_test_fix/offline_chunk_eval.py` 的改动

只有观测历史支持，三处，全部是 patch_policy 的 `n_obs_steps > 1` 逼出来的：

1. `observation.*` 是 `(B, n_obs_steps, ...)`，anchor 取**最后一帧**（部署桥接的起点、`hold_state` 都锚在它上面）。
2. `action` 是 `(B, n_obs_steps - 1 + action_chunk_size, A)`，delta 0 在下标 `n_obs_steps - 1`。
   这是唯一能悄无声息出错的地方——从下标 0 评分就是在跟**过去**比，`check_alignment.py` 守住它。
3. `PatchPolicy.predict_action_chunk` 读的是部署队列而非 batch，批量路径改调 `policy.model.predict`；
   扩散头会采样，所以加了 `--seed` 与 `--seed-repeat`（后者给出采样噪声地板，见报告 §3.4）。

ACT checkpoint 走的仍然是原来的 `predict_action_chunk` 路径，数字与原 harness 可比。
