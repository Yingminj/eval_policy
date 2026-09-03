#!/usr/bin/env bash
# Five checkpoints, one held-out eval set, one harness. ~35 min on a single 4090.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
EVAL=$D/batch_success_53_eval_data
J=/mnt/robot_platform/jobs
COMMON=(--dataset-root "$EVAL" --train-root "$D/batch_success_361"
        --stride 20 --batch-size 8 --num-workers 8
        --n-action-steps 50 --filter-ablation)

run () {  # name ckpt extra...
  local name=$1 ckpt=$2; shift 2
  echo "=== $name"
  $PY offline_chunk_eval.py --checkpoint "$ckpt" "${COMMON[@]}" "$@" \
      --out "$name.json" 2>&1 | tee "$name.log"
}

# the two new weights
run new_state5 \
  $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413/run/checkpoints/200000/pretrained_model \
  --seed-repeat 1
run new_obs2 \
  $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-31-30-522146/run/checkpoints/200000/pretrained_model \
  --seed-repeat 1

# their predecessors (2026-08-20, scored on the same set by scripts_patch_policy_compare)
run prev_diffusion \
  $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model \
  --seed-repeat 1
run prev_act_head \
  $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581/run/checkpoints/200000/pretrained_model

# the yardstick: the ACT checkpoint that actually runs on the robot
run act_baseline \
  $J/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model

echo "ALL DONE"
