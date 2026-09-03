## config

| run | head | n_obs | state | chunk | anchors | eval s |
|---|---|---:|---|---:|---:|---:|
| `new_state5` | diffusion | 5 | True | 50 | 2007 | 332 |
| `new_obs2` | diffusion | 2 | False | 50 | 2007 | 147 |
| `prev_diffusion` | diffusion | 5 | False | 64 | 2007 | 329 |
| `prev_act_head` | act | 5 | False | 50 | 2007 | 39 |
| `act_baseline` | act(policy) | 1 | None | 100 | 2007 | 16 |

## accuracy, executed horizon 8 (raw joint units, MAE)


### policy_raw

| run | @1 | rmse | norm_mae |
|---|---:|---:|---:|
| `new_state5` | 0.03961 | 0.06824 | 0.1277 |
| `new_obs2` | 0.04022 | 0.07013 | 0.1332 |
| `prev_diffusion` | 0.04455 | 0.07761 | 0.1443 |
| `prev_act_head` | 0.04107 | 0.07120 | 0.1333 |
| `act_baseline` | 0.02529 | 0.04981 | 0.0911 |
| *null* `hold_state` | 0.01556 | 0.06273 | 0.0861 |

### policy_deployed

| run | @1 | rmse | norm_mae |
|---|---:|---:|---:|
| `new_state5` | 0.01838 | 0.05964 | 0.0924 |
| `new_obs2` | 0.01781 | 0.06046 | 0.0944 |
| `prev_diffusion` | 0.01883 | 0.06596 | 0.1004 |
| `prev_act_head` | 0.01774 | 0.05882 | 0.0933 |
| `act_baseline` | 0.01652 | 0.04656 | 0.0738 |
| *null* `hold_state` | 0.01556 | 0.06273 | 0.0861 |

## sampling noise (same anchors, second diffusion seed)

| run | seed 0 | seed 1 | spread |
|---|---:|---:|---:|

## deploy filter ladder (Δ vs policy_raw)

| stage | `new_state5` | `new_obs2` | `prev_diffusion` | `prev_act_head` | `act_baseline` |
|---|---:|---:|---:|---:|---:|
| filt_0_clip_only | 0.04214 (+0.0%) | 0.04253 (+0.0%) | 0.04738 (+0.0%) | 0.04336 (-1.5%) | 0.02967 (-0.5%) |
| filt_1_rollbacks | 0.04213 (-0.0%) | 0.04253 (-0.0%) | 0.04738 (+0.0%) | 0.04336 (-1.4%) | 0.02967 (-0.5%) |
| filt_2_gripper_loops | 0.04213 (-0.0%) | 0.04253 (-0.0%) | 0.04738 (+0.0%) | 0.04336 (-1.4%) | 0.02967 (-0.5%) |
| filt_3_smoothing | 0.04096 (-2.8%) | 0.04232 (-0.5%) | 0.04710 (-0.6%) | 0.04336 (-1.4%) | 0.02966 (-0.6%) |
| filt_4_excursions | 0.04096 (-2.8%) | 0.04232 (-0.5%) | 0.04710 (-0.6%) | 0.04336 (-1.4%) | 0.02966 (-0.6%) |
| filt_5_bridge | 0.03107 (-26.3%) | 0.03084 (-27.5%) | 0.03351 (-29.3%) | 0.03103 (-29.5%) | 0.02430 (-18.5%) |
| filt_bridge_only | 0.03107 (-26.3%) | 0.03084 (-27.5%) | 0.03351 (-29.3%) | 0.03103 (-29.5%) | 0.02430 (-18.5%) |

## per-horizon MAE, policy_raw (first 50)

new_state5: 0.0396 0.0433
new_obs2: 0.0402 0.0427
prev_diffusion: 0.0445 0.0478
prev_act_head: 0.0411 0.0444
act_baseline: 0.0253 0.0305
cuts: step 1, 5, 10, 20, 30, 40, 50

## gripper vs arm (policy_raw, per-joint MAE)

| run | arm joints (mean) | gripper_L | gripper_R |
|---|---:|---:|---:|
| `new_state5` | 0.04430 | 0.02256 | 0.03156 |
| `new_obs2` | 0.04550 | 0.02276 | 0.02076 |
| `prev_diffusion` | 0.04978 | 0.02583 | 0.03529 |
| `prev_act_head` | 0.04665 | 0.02366 | 0.02727 |
| `act_baseline` | 0.03214 | 0.01231 | 0.01488 |
