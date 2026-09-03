#!/usr/bin/env bash
# The 08-31 report's mechanism claim, re-measured on the independent eval set and for ACT too.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
run () { local n=$1 ck=$2 root=$3; echo "=== $n"; $PY anchor.py "$ck" "$root" --out "anchor_$n.json" 2>&1 | tail -25 | tee "anchor_$n.log"; }

run acteef_505 $J/act_eef_tidy_up_stationery_le_batch_success_505_eef/checkpoints/200000/pretrained_model $D/batch_success_53_eval_data_eef
run act_joint_361 $J/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model $D/batch_success_53_eval_data
run pp_joint_state5 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413/run/checkpoints/200000/pretrained_model $D/batch_success_53_eval_data
run pp_eef $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints/200000/pretrained_model $D/batch_success_53_eval_data_eef
echo ANCHOR DONE
