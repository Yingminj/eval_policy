
# EEF space, horizon 50 unless noted

## config

| run | type | head | n_obs | state | chunk | train set | anchors | eps | s |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| `pp_eef_200k` | patch_policy | diffusion | 2 | True | 50 | `batch_success_505_eef` | 2007 | 53 | 278 |
| `pp_eef_100k` | patch_policy | diffusion | 2 | True | 50 | `batch_success_505_eef` | 2007 | 53 | 142 |
| `acteef_505_200k` | act_eef | - | 1 | None | 100 | `batch_success_505_eef` | 2007 | 53 | 11 |
| `acteef_505_100k` | act_eef | - | 1 | None | 100 | `batch_success_505_eef` | 2007 | 53 | 11 |
| `acteef_505_h60` | act_eef | - | 1 | None | 100 | `batch_success_505_eef` | 2007 | 53 | 11 |
| `acteef_361_200k` | act_eef | - | 1 | None | 100 | `batch_success_361_eef` | 2007 | 53 | 11 |
| `acteef_533_200k` | act_eef | - | 1 | None | 100 | `batch_success_533_eef` | 2007 | 53 | 12 |

## accuracy (MAE over metres + radians + [0,1] gripper -- read the group table)

| run | eval set | @1 | @10 | @25 | @50 | rmse | norm_mae | vs null |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_200k` | 53-ep independent | 0.02254 | 0.02564 | 0.02994 | 0.03721 | 0.07869 | 0.1879 | 1.95x |
| `pp_eef_100k` | 53-ep independent | 0.03563 | 0.03637 | 0.03866 | 0.04394 | 0.08574 | 0.2196 | 1.65x |
| `acteef_505_200k` | 53-ep independent | 0.01814 | 0.02273 | 0.02892 | 0.03689 | 0.07902 | 0.1817 | 1.97x |
| `acteef_505_100k` | 53-ep independent | 0.01891 | 0.02318 | 0.02922 | 0.03704 | 0.07951 | 0.1826 | 1.96x |
| `acteef_505_h60` | 53-ep independent | 0.01814 | 0.02273 | 0.02892 | 0.03689 | 0.08481 | 0.1950 | 2.07x |
| `acteef_361_200k` | 53-ep independent | 0.01924 | 0.02353 | 0.02889 | 0.03568 | 0.07465 | 0.1777 | 2.04x |
| `acteef_533_200k` | 53-ep independent | 0.01694 | 0.02129 | 0.02713 | 0.03443 | 0.07360 | 0.1715 | 2.11x |
| `pp_eef_200k (08-31)` | 67-ep random (08-31) | 0.02170 | 0.02443 | 0.02816 | 0.03323 | 0.07548 | 0.1699 | 2.00x |
| `acteef_361 (08-31 fair)` | 18-ep random (08-31) | 0.03267 | 0.03782 | 0.04552 | 0.05453 | 0.11366 | 0.2697 | 1.27x |
| *null* `hold_state` | 53-ep independent | 0.02098 | 0.03138 | 0.04757 | 0.07266 | 0.17191 | 0.3640 | - |
| *null* `hold_state` | 67-ep random (08-31) | 0.02283 | 0.03131 | 0.04518 | 0.06649 | 0.16578 | 0.3342 | - |
| *null* `hold_state` | 18-ep random (08-31) | 0.02744 | 0.03567 | 0.04871 | 0.06933 | 0.16756 | 0.3518 | - |

### sampling noise (same anchors, second diffusion seed)

| run | seed 0 | seed 1 | spread |
|---|---:|---:|---:|
| `pp_eef_200k` | 0.03721 | 0.03757 | +1.0% |
| `pp_eef_200k (08-31)` | 0.03323 | 0.03362 | +1.2% |

## per-group MAE

| run | eval set | position (m) | rotation (rad) | gripper (0-1) |
|---|---|---:|---:|---:|
| `pp_eef_200k` | 53-ep independent | 0.01128 | 0.06736 | 0.02454 |
| `pp_eef_100k` | 53-ep independent | 0.01320 | 0.07846 | 0.03261 |
| `acteef_505_200k` | 53-ep independent | 0.01023 | 0.06784 | 0.02401 |
| `acteef_505_100k` | 53-ep independent | 0.01044 | 0.06738 | 0.02586 |
| `acteef_505_h60` | 53-ep independent | 0.01095 | 0.07284 | 0.02587 |
| `acteef_361_200k` | 53-ep independent | 0.01026 | 0.06560 | 0.02219 |
| `acteef_533_200k` | 53-ep independent | 0.00978 | 0.06296 | 0.02284 |
| `pp_eef_200k (08-31)` | 67-ep random (08-31) | 0.01060 | 0.05817 | 0.02632 |
| `acteef_361 (08-31 fair)` | 18-ep random (08-31) | 0.01558 | 0.10051 | 0.03340 |
| `*null* hold_state (53-ep independent)` | 53-ep independent | 0.02362 | 0.12323 | 0.06807 |
| `*null* hold_state (67-ep random (08-31))` | 67-ep random (08-31) | 0.02154 | 0.11413 | 0.05843 |
| `*null* hold_state (18-ep random (08-31))` | 18-ep random (08-31) | 0.02235 | 0.11914 | 0.06082 |

# pose anchoring: does the chunk start where the arm is?

| run | space | policy \|chunk[0]-state\| | demo \|action[0]-state\| | ratio |
|---|---|---:|---:|---:|
| `pp_eef` | eef | 0.02936 | 0.02098 | 1.399 |
| `acteef_505` | eef | 0.02823 | 0.02098 | 1.345 |
| `pp_joint_state5` | joint | 0.04144 | 0.01556 | 2.663 |
| `act_joint_361` | joint | 0.02982 | 0.01556 | 1.916 |

Units are the dataset's own (joint radians / metres+radians+gripper), so compare the ratio column across spaces and the absolute columns only within one.

# same physical quantity: every chunk pushed through the dataset's own FK

| run | predicts in | position mm | rotation deg | gripper | @1 | @10 | @25 | @50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_200k` | eef | 11.28 | 3.860 | 0.02454 | 0.02254 | 0.02564 | 0.02994 | 0.03721 |
| `acteef_505_200k` | eef | 10.23 | 3.887 | 0.02401 | 0.01814 | 0.02273 | 0.02892 | 0.03689 |
| `act_joint_361` | joint | 11.28 | 3.743 | 0.02268 | 0.02234 | 0.02549 | 0.02997 | 0.03607 |
| `pp_joint_state5` | joint | 15.55 | 4.848 | 0.03375 | 0.04143 | 0.03987 | 0.04261 | 0.04775 |
| *null* `hold_state` (via the eef eval set) | - | 23.62 | 7.061 | 0.06807 | 0.02098 | 0.03138 | 0.04757 | 0.07266 |
| *null* `hold_state` (via the joint eval set) | - | 23.62 | 7.061 | 0.06807 | 0.02098 | 0.03138 | 0.04757 | 0.07266 |

The two `hold_state` rows must agree: the joint and EEF eval sets are the same recordings and FK is deterministic, so 'the arm stays put' is the same trajectory in both. A gap there means the anchors do not line up.

# joint-space reference (unchanged, from ../scripts_patch_policy_eval_fix)

| run | eval set | @1 | @10 | @25 | @50 | rmse | norm_mae | vs null |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `new_state5` | 53-ep independent, joint space | 0.03961 | 0.04277 | 0.04968 | 0.06121 | 0.10561 | 0.1886 | 1.59x |
| `act_baseline` | 53-ep independent, joint space | 0.02529 | 0.03115 | 0.03972 | 0.05112 | 0.09132 | 0.1575 | 1.90x |
| *null* `hold_state` | 53-ep independent, joint space | 0.01556 | 0.03174 | 0.05721 | 0.09705 | 0.20211 | 0.2988 | - |

| run | eval set | arm joints | gripper_L | gripper_R |
|---|---|---:|---:|---:|
| `new_state5` | 53-ep independent, joint space | 0.06514 | 0.03102 | 0.03648 |
| `act_baseline` | 53-ep independent, joint space | 0.05518 | 0.02220 | 0.02315 |
| `*null* hold_state (53-ep independent, joint space)` | 53-ep independent, joint space | 0.10119 | 0.06813 | 0.06801 |
