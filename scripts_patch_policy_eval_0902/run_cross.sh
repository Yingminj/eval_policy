#!/usr/bin/env bash
# Joint- and EEF-space policies scored on the same physical quantity (see cross_space.py).
# Same 53 recordings, same stride, so the anchors line up across the two datasets.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs

run () { local n=$1 ck=$2 root=$3; shift 3; echo "=== $n"; \
  $PY cross_space.py "$n" "$ck" "$root" "$@" --out "cross_$n.json" 2>&1 | tee "cross_$n.log"; }

run acteef_505_200k $J/act_eef_tidy_up_stationery_le_batch_success_505_eef/checkpoints/200000/pretrained_model \
    $D/batch_success_53_eval_data_eef
run act_joint_361 $J/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model \
    $D/batch_success_53_eval_data
run pp_joint_state5 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413/run/checkpoints/200000/pretrained_model \
    $D/batch_success_53_eval_data
run pp_eef_200k $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints/200000/pretrained_model \
    $D/batch_success_53_eval_data_eef
echo "CROSS DONE"
