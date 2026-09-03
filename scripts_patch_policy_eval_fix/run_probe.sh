#!/usr/bin/env bash
# What each checkpoint conditions on. ~2 min per checkpoint (48 anchors x 10 interventions).
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
for pair in \
  "new_state5 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413" \
  "new_obs2 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-31-30-522146" \
  "prev_act_head $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581"
do
  set -- $pair
  echo "=== probe $1"
  $PY probe_conditioning.py --checkpoint "$2/run/checkpoints/200000/pretrained_model" \
    --dataset-root "$D/batch_success_53_eval_data" --n-anchors 96 --stride 401 \
    --out "probe_$1.json" 2>&1 | tee "probe_$1.log"
done
echo "PROBE DONE"
