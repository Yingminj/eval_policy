#!/usr/bin/env bash
# Final GPU queue: deployed-trajectory evals, then the two one-field ablation arms.
set -u
PY=/opt/robot-platform/train-venv/bin/python
CKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
ACTCKPT=/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581/run/checkpoints/200000/pretrained_model
DS=/mnt/robot_platform/datasets/tidy_up_stationery_le
cd "$(dirname "$0")"

while [ ! -f patch_intrain_reanchored.json ]; do sleep 30; done
echo "=== queue3 start $(date) ==="

echo "--- deployed trajectory, held-out ---"
$PY eval_deploy_bridge.py --checkpoint $CKPT --dataset-root $DS/batch_1 --dataset-root $DS/batch_2 \
  --dataset-root $DS/batch_3 --dataset-root $DS/batch_4 --train-root $DS/batch_success_361 \
  --stride 20 --out bridge_heldout.json > bridge_heldout.log 2>&1

echo "--- deployed trajectory, in-training ---"
$PY eval_deploy_bridge.py --checkpoint $CKPT --dataset-root $DS/batch_success_361 \
  --stride 100 --out bridge_intrain.json > bridge_intrain.log 2>&1

echo "--- deployed trajectory, ACT head, held-out ---"
$PY eval_deploy_bridge.py --checkpoint $ACTCKPT --dataset-root $DS/batch_3 \
  --train-root $DS/batch_success_361 --stride 20 --out bridge_acthead_heldout.json \
  > bridge_acthead_heldout.log 2>&1

for arm in "shipped:" "state:use_robot_state=True"; do
  name="${arm%%:*}"; ov="${arm#*:}"
  echo "--- ablation $name ($(date)) ---"
  args=(--name "$name" --base-config $CKPT --dataset-root $DS/batch_success_361 \
        --heldout-root $DS/batch_3 --train-root $DS/batch_success_361 \
        --steps 20000 --val-every 2500 --val-anchors 128 --waypoint 8 --out "abl_${name}.json")
  [ -n "$ov" ] && args+=(--override "$ov")
  $PY train_patch_ablation.py "${args[@]}" > "abl_${name}.log" 2>&1
done

echo "=== queue3 done $(date) ==="
