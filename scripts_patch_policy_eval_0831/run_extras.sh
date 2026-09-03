#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs

echo "=== vq_floor"
$PY vq_floor.py \
  --checkpoint $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-19-40-151329/run/checkpoints/100000/pretrained_model \
  --dataset-root $D/batch_success_53_eval_data --stride 20 \
  --out vq_floor.json 2>&1 | tee vq_floor.log

echo "=== latency (needs an idle GPU)"
$PY latency.py --out latency.json 2>&1 | tee latency.log
echo "EXTRAS DONE"
