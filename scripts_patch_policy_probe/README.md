# `patch_policy` diagnostic scripts

Four scripts, all run with `/opt/robot-platform/train-venv/bin/python` (the interpreter that
trained the checkpoints). They import the contamination fingerprinting, the error accumulator and
the `hold_state` / `train_mean` baselines from `../scripts_act_eval_test/offline_chunk_eval.py`,
so the numbers here are produced by the same code as the ACT and ACT-DiT reports and are directly
comparable to them.

| script | question | cost |
|---|---|---|
| `eval_patch_chunk.py` | how far is the emitted chunk from the demonstration, on episodes the model never saw? | ~0.2 s/anchor |
| `probe_patch_conditioning.py` | which input actually moves the output — each camera, the history, the dense patch grid? | ~2 min |
| `probe_state_decodability.py` | can the arm's pose be read out of the frozen encoder at all? | ~10 min |
| `train_patch_ablation.py` | does changing one config field fix it, scored on held-out data? | ~4 min / 1000 steps |

`run_queue.sh` chains the first three; they all want the whole GPU, so nothing in it overlaps.

## Why these are separate from the ACT harness

`offline_chunk_eval.py` cannot score `patch_policy` unmodified, for three reasons that are
structural rather than cosmetic:

* the chunk length is `action_chunk_size`, not `chunk_size`;
* `n_obs_steps = 5`, so observations arrive as `(B, T, ...)` and the dataloader delivers
  `n_obs_steps + action_chunk_size - 1` actions. The predicted chunk lines up with
  `action[:, n_obs_steps - 1:]` — scoring it against `action[:, 0:]` silently compares the chunk
  to a window starting four frames in the past;
* `predict_action_chunk` reads the online queues and slices to `n_action_steps`, so the offline
  path calls `policy.model.predict` directly and keeps the whole horizon.

## What the baselines mean

* **`hold_state`** — emit the current measured joint pose for the whole chunk, i.e. do nothing.
  A policy that cannot beat this has not learned the task, whatever its training loss says.
* **`train_mean`** — emit the training-set mean action. The floor.
* **`policy_reanchored`** — *not* a baseline: it is closer to the deployed policy than `policy` is.
  `deploy_config_patch_policy.yaml` sets `n_action_steps: 8`, and
  `rollout/strategies/core.py::send_next_action_chunk` then sets `bridge_steps = min(40, 8) = 8`,
  so **every arm-joint frame sent to the robot is a cubic Hermite from the measured pose to
  `chunk[7]`** (the two gripper channels are excluded from the bridge and pass through raw).
  Re-anchoring the predicted chunk onto the measured pose reproduces that translation. The gap
  between `policy` and `policy_reanchored` separates *"does not know where the arm is"* from
  *"does not know what motion to make"* — two failures with completely different fixes.

## Scale references for reading any of these numbers

Measured on this corpus, in raw joint units (rad; the two gripper channels are 0–1):

| quantity | value |
|---|---|
| `|action_t - state_t|` (the action is a near-copy of the current pose) | 0.012 |
| `|action_{t+8} - state_t|` (the whole motion one deploy waypoint covers) | 0.032 |
| within-episode `std(state)` | 0.324 |
| gripper frames with a transition | 4.5 % |

A waypoint error above ~0.032 rad means the command carries no motion information: the error is
larger than the movement being asked for.

## Running them

```bash
PY=/opt/robot-platform/train-venv/bin/python
CKPT=/mnt/robot_platform/jobs/patch_policy_..._2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
DS=/mnt/robot_platform/datasets/tidy_up_stationery_le

# held-out chunk error (the contamination filter is not optional -- batch_5/6 are training data)
$PY eval_patch_chunk.py --checkpoint $CKPT \
    --dataset-root $DS/batch_1 --dataset-root $DS/batch_2 \
    --dataset-root $DS/batch_3 --dataset-root $DS/batch_4 \
    --train-root $DS/batch_success_361 --stride 20 --out patch_heldout.json

# what the policy conditions on
$PY probe_patch_conditioning.py --checkpoint $CKPT --dataset-root $DS/batch_3 \
    --train-root $DS/batch_success_361 --out probe_conditioning_heldout.json

# whether the frozen encoder carries the arm pose at all
$PY probe_state_decodability.py --checkpoint $CKPT --train-root $DS/batch_success_361 \
    --heldout-root $DS/batch_3 --out probe_state_decodability.json

# one-field ablations, scored on held-out data every --val-every steps
$PY train_patch_ablation.py --name state --override use_robot_state=True \
    --base-config $CKPT --dataset-root $DS/batch_success_361 \
    --heldout-root $DS/batch_3 --train-root $DS/batch_success_361 \
    --steps 20000 --out abl_state.json
```

`train_patch_ablation.py` takes its base config from a real checkpoint directory, not from
`PatchPolicyConfig()` defaults, so the `shipped` arm is genuinely what shipped; `--override`
rejects unknown field names, because a typo'd override would otherwise reproduce `shipped`
exactly and be written up as a null result.

Self-checks: `train_patch_ablation.py --selftest` (override handling).
