#!/usr/bin/env bash
# What each of the two new checkpoints conditions on. ~2-4 min per checkpoint.
# Each is probed on the eval set of its OWN action space.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs

echo "=== probe vqbet_100k"
$PY probe_conditioning.py \
  --checkpoint $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-19-40-151329/run/checkpoints/100000/pretrained_model \
  --dataset-root $D/batch_success_53_eval_data --n-anchors 96 --stride 401 \
  --out probe_vqbet_100k.json 2>&1 | tee probe_vqbet_100k.log

echo "=== probe eef_200k"
$PY probe_conditioning.py \
  --checkpoint $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints/200000/pretrained_model \
  --dataset-root $D/batch_success_533_eef --train-root $D/batch_success_505_eef \
  --n-anchors 96 --stride 401 \
  --out probe_eef_200k.json 2>&1 | tee probe_eef_200k.log
echo "PROBE DONE"
