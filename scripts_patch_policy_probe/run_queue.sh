#!/usr/bin/env bash
# Serialised GPU queue: everything here needs the whole 4090, so nothing overlaps.
set -u
PY=/opt/robot-platform/train-venv/bin/python
CKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
ACTCKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581/run/checkpoints/200000/pretrained_model
DS=/mnt/robot_platform/datasets/tidy_up_stationery_le
cd "$(dirname "$0")"

# NB: not `pgrep -f eval_patch_chunk.py` -- the shell wrappers that launched the eval keep
# that string in their own cmdline forever, so pgrep never goes quiet. Wait on the output.
while [ ! -f patch_heldout.json ]; do sleep 30; done
echo "=== queue start $(date) ==="

echo "--- conditioning probe (held-out) ---"
$PY probe_patch_conditioning.py --checkpoint $CKPT --dataset-root $DS/batch_3 \
  --train-root $DS/batch_success_361 --n-anchors 48 --out probe_conditioning_heldout.json \
  > probe_conditioning_heldout.log 2>&1

echo "--- state decodability ---"
$PY probe_state_decodability.py --checkpoint $CKPT --train-root $DS/batch_success_361 \
  --heldout-root $DS/batch_3 --out probe_state_decodability.json \
  > probe_state_decodability.log 2>&1

echo "--- ACT-head checkpoint, held-out chunk eval (control for the head) ---"
$PY eval_patch_chunk.py --checkpoint $ACTCKPT \
  --dataset-root $DS/batch_3 --train-root $DS/batch_success_361 --stride 20 --batch-size 16 \
  --out patch_acthead_heldout.json > patch_acthead_heldout.log 2>&1

echo "=== queue done $(date) ==="
