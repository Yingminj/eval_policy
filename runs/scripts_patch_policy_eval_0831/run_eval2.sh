#!/usr/bin/env bash
# Two controls the main sweep does not cover.
#
#  *_intrain   the SAME checkpoint on the episodes it was trained on.  The held-out set is
#              a random interleaved split, so "policy vs null = 2.0x" could be memorisation
#              of neighbouring frames as easily as generalisation; the in-train/held-out gap
#              is what separates them.
#  vqbet_*_intrain  the same question for the joint-space weight, on the same eval corpus
#              the 08-30 report used, so the two gaps are measured the same way.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
EEF=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints
VQ=$J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-19-40-151329/run/checkpoints

run () { local name=$1; shift; echo "=== $name"; $PY offline_chunk_eval.py "$@" --out "$name.json" 2>&1 | tee "$name.log"; }

# Same anchor budget as the held-out run (2814) so the two are the same size.
run eef_200k_intrain --checkpoint "$EEF/200000/pretrained_model" \
  --dataset-root "$D/batch_success_533_eef" --train-root "$D/batch_success_505_eef" \
  --keep-only-contaminated --stride 160 --max-anchors-per-dataset 2814 \
  --batch-size 8 --num-workers 8 --n-action-steps 50 --filters none

run vqbet_100k_intrain --checkpoint "$VQ/100000/pretrained_model" \
  --dataset-root "$D/batch_success_361" --train-root "$D/batch_success_361" \
  --keep-only-contaminated --stride 150 --max-anchors-per-dataset 2007 \
  --batch-size 8 --num-workers 8 --n-action-steps 50 --filter-ablation
echo "ALL DONE 2"
