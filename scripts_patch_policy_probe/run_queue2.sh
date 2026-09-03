#!/usr/bin/env bash
# Second GPU queue: starts when run_queue.sh reports done. Nothing here overlaps either.
set -u
PY=/opt/robot-platform/train-venv/bin/python
CKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
DS=/mnt/robot_platform/datasets/tidy_up_stationery_le
cd "$(dirname "$0")"

while ! grep -q "queue done" run_queue.log 2>/dev/null; do sleep 30; done
echo "=== queue2 start $(date) ==="

# The first in-training pass predates the `policy_reanchored` metric, and that metric is the
# one that separates a pose-localisation failure from a trajectory failure. Re-run it coarser.
echo "--- in-training, re-anchored (stride 100) ---"
$PY eval_patch_chunk.py --checkpoint $CKPT --dataset-root $DS/batch_success_361 \
  --stride 100 --batch-size 16 --out patch_intrain_reanchored.json \
  > patch_intrain_reanchored.log 2>&1

for arm in "shipped:" "state:use_robot_state=True"; do
  name="${arm%%:*}"; ov="${arm#*:}"
  echo "--- ablation $name ($(date)) ---"
  args=(--name "$name" --base-config $CKPT --dataset-root $DS/batch_success_361 \
        --heldout-root $DS/batch_3 --train-root $DS/batch_success_361 \
        --steps 20000 --val-every 2500 --val-anchors 128 --waypoint 8 --out "abl_${name}.json")
  [ -n "$ov" ] && args+=(--override "$ov")
  $PY train_patch_ablation.py "${args[@]}" > "abl_${name}.log" 2>&1
done

echo "=== queue2 done $(date) ==="
