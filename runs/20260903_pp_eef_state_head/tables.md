eval set: 53 episodes / 40132 frames / 2007 anchors / 97090 scored action steps, stride 20, contaminated episodes dropped: 0
deploy filters: ['rollbacks', 'smoothing', 'bridge', 'gripper_clip']   batch_size 8   seed 0

### 1. main table — executed window (`--n-action-steps 60`, clamped to each policy's chunk)

| run | head | n_obs | state | horizon | raw | deployed | @1 | @10 | @25 | @50 | raw vs null | deployed vs null | 墙钟 (s) | 每 anchor (ms) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_state (08-31)` | diffusion | 2 | on | 50 |  0.03721 |  0.04034 |  0.02254 |  0.02564 |  0.02994 |  0.03721 | 1.95x | 1.80x | 286 (2 draws) | 71.2 |
| `pp_eef_nostate (09-44)` | diffusion | 2 | off | 50 |  0.03916 |  0.04090 |  0.02613 |  0.02877 |  0.03271 |  0.03916 | 1.86x | 1.78x | 288 (2 draws) | 71.7 |
| `pp_eef_act5 (09-33)` | act | 5 | off | 50 |  0.03796 |  0.03912 |  0.02564 |  0.02840 |  0.03222 |  0.03796 | 1.91x | 1.86x | 56 (2 draws) | 13.9 |
| `acteef_533 (deployed)` | act(ACT) | 1 | off | 60 |  0.03689 |  0.03975 |  0.01694 |  0.02129 |  0.02713 |  0.03443 | 2.23x | 2.07x | 16 | 8.2 |

| null baseline | mae | @1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|---:|
| `hold_state` @ horizon 50 |  0.07266 |  0.02098 |  0.03138 |  0.04757 |  0.07266 |
| `train_mean` @ horizon 50 |  0.22729 |  0.22632 |  0.22650 |  0.22676 |  0.22729 |
| `hold_state` @ horizon 60 |  0.08216 |  0.02098 |  0.03138 |  0.04757 |  0.07266 |
| `train_mean` @ horizon 60 |  0.22685 |  0.22570 |  0.22587 |  0.22614 |  0.22668 |

**相对基线 `pp_eef_state` 的差值**（同一 anchor 集、同一 seed、同一 batch size）：

| run | raw Δ | deployed Δ | @1 Δ | 位置 Δ | 采样噪声下限 |
|---|---:|---:|---:|---:|---:|
| `pp_eef_nostate (09-44)` | +5.2% | +1.4% | +15.9% | +8.7% | 1.58% |
| `pp_eef_act5 (09-33)` | +2.0% | -3.0% | +13.8% | +0.9% | 0（确定性） |
| `acteef_533 (deployed)` **(h60，跨 horizon，不可直接读，见 §7)** | -0.9% | -1.5% | -24.9% | -6.9% | 0（确定性） |

`pp_eef_act5` vs `pp_eef_nostate`（两者 `use_robot_state` 都是 off，差的是 head 与 n_obs）：raw -3.1%, deployed -4.4%, 墙钟 0.19x

### 2. rmse and tail

| run | raw rmse | deployed rmse | norm_mae | norm_rmse | tail ratio |
|---|---:|---:|---:|---:|---:|
| `pp_eef_state (08-31)` |  0.07869 |  0.08944 |  0.18789 |  0.30994 | 1.65 |
| `pp_eef_nostate (09-44)` |  0.08025 |  0.08922 |  0.20095 |  0.32147 | 1.60 |
| `pp_eef_act5 (09-33)` |  0.07607 |  0.08555 |  0.19238 |  0.30859 | 1.60 |
| `acteef_533 (deployed)` |  0.07867 |  0.08788 |  0.18406 |  0.31150 | 1.69 |

### 3. grouped error — the 14 dims do not share a unit

| run | 位置 MAE (mm) | 姿态 MAE (°) | 夹爪 MAE (0-1) | 夹爪 clip 后 | 位置 deployed (mm) | 姿态 deployed (°) |
|---|---:|---:|---:|---:|---:|---:|
| `pp_eef_state (08-31)` | 11.28 | 3.86 | 0.0245 | 0.0245（未越界） | 12.52 | 4.21 |
| `pp_eef_nostate (09-44)` | 12.27 | 4.05 | 0.0255 | 0.0255（未越界） | 13.00 | 4.24 |
| `pp_eef_act5 (09-33)` | 11.38 | 3.93 | 0.0260 | 0.0217 | 12.27 | 4.11 |
| `acteef_533 (deployed)` | 10.51 | 3.87 | 0.0243 | 0.0214 | 11.98 | 4.22 |
| *null* `hold_state` (h50) | 23.62 | 7.06 | 0.0681 | — | — | — |
| *null* `train_mean` (h50) | 45.43 | 18.64 | 0.4789 | — | — | — |

### 4. sampling noise — is any of the above bigger than the head's own variance?

| run | seed 0 (raw) | seed 1 | Δ | 确定性? |
|---|---:|---:|---:|---|
| `pp_eef_state (08-31)` |  0.03721 |  0.03757 | +0.363e-3 | 否（0.97%） |
| `pp_eef_nostate (09-44)` |  0.03916 |  0.03854 | -0.617e-3 | 否（1.58%） |
| `pp_eef_act5 (09-33)` |  0.03796 |  0.03796 | +0.000e-3 | **是**（逐位相同） |
| `acteef_533 (deployed)` |  0.03689 | — | — | 未测（ACT，无采样） |

### 5. deploy filter ladder (cumulative, deploy order)

`gripper_loops` and `excursions` are ENABLE_*=False on the robot, so their rungs repeat the previous one by construction.

| run | raw | 0_clip_only | 1_rollbacks | 2_gripper_loops | 3_smoothing | 4_excursions | 5_bridge | bridge_only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_state (08-31)` | 0.03721 | 0.03721 | 0.03719 | 0.03719 | 0.03709 | 0.03709 | 0.04034 | 0.04081 |
| `pp_eef_nostate (09-44)` | 0.03916 | 0.03916 | 0.03915 | 0.03915 | 0.03905 | 0.03905 | 0.04090 | 0.04144 |
| `pp_eef_act5 (09-33)` | 0.03796 | 0.03735 | 0.03735 | 0.03735 | 0.03735 | 0.03735 | 0.03912 | 0.03931 |
| `acteef_533 (deployed)` | 0.03689 | 0.03648 | 0.03648 | 0.03648 | 0.03646 | 0.03646 | 0.03975 | 0.03985 |

相对 `policy_raw` 的百分比：

| run | 0_clip_only | 1_rollbacks | 2_gripper_loops | 3_smoothing | 4_excursions | 5_bridge | bridge_only |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_state (08-31)` | +0.0% | -0.1% | -0.1% | -0.3% | -0.3% | +8.4% | +9.7% |
| `pp_eef_nostate (09-44)` | +0.0% | -0.0% | -0.0% | -0.3% | -0.3% | +4.5% | +5.8% |
| `pp_eef_act5 (09-33)` | -1.6% | -1.6% | -1.6% | -1.6% | -1.6% | +3.1% | +3.6% |
| `acteef_533 (deployed)` | -1.1% | -1.1% | -1.1% | -1.2% | -1.2% | +7.8% | +8.0% |

### 6. 100k vs 200k — is either new arm still improving?

| run | 100k raw | 200k raw | Δ | 100k deployed | 200k deployed |
|---|---:|---:|---:|---:|---:|
| `pp_eef_nostate (09-44)` |  0.04649 |  0.03916 | -15.8% |  0.04445 |  0.04090 |
| `pp_eef_act5 (09-33)` |  0.03778 |  0.03796 | +0.5% |  0.03909 |  0.03912 |

### 7. horizon 50 vs 60 for the ACT baseline (patch_policy cannot reach 60)

| run | horizon | chunk | raw | deployed | hold_state | raw vs null | deployed vs null | 位置 (mm) | 姿态 (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `acteef_533_h50` | 50 | 100 |  0.03443 |  0.03794 |  0.07266 | 2.11x | 1.92x | 9.78 | 3.61 |
| `acteef_533_h60` | 60 | 100 |  0.03689 |  0.03975 |  0.08216 | 2.23x | 2.07x | 10.51 | 3.87 |

selftest OK: published pp_eef/acteef_533 numbers reproduce; act head is deterministic
