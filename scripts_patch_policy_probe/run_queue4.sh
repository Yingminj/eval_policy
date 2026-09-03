#!/usr/bin/env bash
# Fourth queue: the stronger-injection arm, once the `state` arm frees the GPU.
set -u
PY=/opt/robot-platform/train-venv/bin/python
CKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
DS=/mnt/robot_platform/datasets/tidy_up_stationery_le
cd "$(dirname "$0")"

while [ ! -f abl_state.json ]; do sleep 60; done
echo "=== queue4 start $(date) ==="
$PY train_patch_ablation.py --name state_broadcast --override use_robot_state=True --state-broadcast \
  --base-config $CKPT --dataset-root $DS/batch_success_361 --heldout-root $DS/batch_3 \
  --train-root $DS/batch_success_361 --steps 20000 --val-every 2500 --val-anchors 128 --waypoint 8 \
  --out abl_state_broadcast.json > abl_state_broadcast.log 2>&1
echo "=== queue4 done $(date) ==="
