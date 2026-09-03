#!/usr/bin/env bash
# The same five checkpoints at the horizon patch_policy is ACTUALLY deployed with:
# deploy_config_patch_policy.yaml sets inference.n_action_steps: 8, and the deploy bridge
# is min(40, len(chunk)) -- so at 8 the whole executed window is the Hermite S-curve.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
COMMON=(--dataset-root "$D/batch_success_53_eval_data" --train-root "$D/batch_success_361"
        --stride 20 --batch-size 8 --num-workers 8 --n-action-steps 8 --filter-ablation)

run () {
  local name=$1 ckpt=$2
  echo "=== $name"
  $PY offline_chunk_eval.py --checkpoint "$ckpt" "${COMMON[@]}" --out "${name}_h8.json" 2>&1 | tee "${name}_h8.log"
}

run new_state5 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413/run/checkpoints/200000/pretrained_model
run new_obs2 $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-31-30-522146/run/checkpoints/200000/pretrained_model
run prev_diffusion $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model
run prev_act_head $J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-21-58-821581/run/checkpoints/200000/pretrained_model
run act_baseline $J/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model
echo "ALL DONE H8"
