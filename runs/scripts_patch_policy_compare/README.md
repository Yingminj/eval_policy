# scripts_patch_policy_compare

`../../patch_policy/patch_policy-head-comparison-2026-08.md` 的测量脚本与原始结果。

| 文件 | 作用 | 报告章节 |
|---|---|---|
| `contam.py` → `contam.json` | 评测集/训练集 episode 去重核查 | §3.2 |
| `eval_patch.py` → `results.json`, `eval.log` | 精度（1606 anchor）+ 9 种视觉消融 | §4, §7 |
| `sweep.py` → `sweep.json`, `sweep.log` | checkpoint 扫描、re-anchor、量纲参照 | §5, §6 |
| `latency.py` → `latency.json` | batch-1 推理延迟（独占 GPU） | §8 |

解释器必须是 `/opt/robot-platform/train-venv/bin/python`（训练这两个 checkpoint 的同一环境）。

```bash
V=/opt/robot-platform/train-venv/bin/python
$V contam.py
$V eval_patch.py results.json 25   # ~45 min，其中 2497 s 花在 B 的 100 步 DDPM 上
$V sweep.py                        # ~12 min
$V latency.py                      # ~1 min，跑之前确认 GPU 无其他进程
```

`sweep.log` 里的 `latency` 两行是**污染数据**（首轮有两个进程并发占用 GPU），
以 `latency.json` 为准。精度数字是定种子的，不受影响，两轮一致。
