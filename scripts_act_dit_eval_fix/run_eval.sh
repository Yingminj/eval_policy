#!/usr/bin/env bash
# ACT-DiT (flow matching + EMA, 2026-08-27 run) under the deployment-faithful harness.
set -euo pipefail
S=/home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_act_eval_test_fix
O=/home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_act_dit_eval_fix
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
PY=/opt/robot-platform/train-venv/bin/python
CKPT=${CKPT:-/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-27_04-32-02-437338/run/checkpoints/last/pretrained_model}
TAG=${TAG:-dit}
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src   # act_dit EMA lives here; the train-venv copy predates it
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
cd "$S"
common=(--checkpoint "$CKPT" --stride 20 --batch-size 32 --num-workers 8 --n-action-steps 50 --filter-ablation)

echo "### eval53 (designated eval set)"; time $PY offline_chunk_eval.py "${common[@]}" \
  --train-root $D/batch_success_361 --dataset-root $D/batch_success_53_eval_data \
  --out $O/eval53_${TAG}_deployed.json

echo "### in-training control"; time $PY offline_chunk_eval.py "${common[@]}" \
  --dataset-root $D/batch_success_361 --out $O/intrain_control_${TAG}_deployed.json

echo "### held-out batch_1-6, contamination filtered"; time $PY offline_chunk_eval.py "${common[@]}" \
  --train-root $D/batch_success_361 \
  --dataset-root $D/batch_1 --dataset-root $D/batch_2 --dataset-root $D/batch_3 \
  --dataset-root $D/batch_4 --dataset-root $D/batch_5 --dataset-root $D/batch_6 \
  --out $O/heldout_clean_${TAG}_deployed.json

echo "### within-session control: seen half of batch_4"; time $PY offline_chunk_eval.py "${common[@]}" \
  --train-root $D/batch_success_361 --dataset-root $D/batch_4 --keep-only-contaminated \
  --out $O/within_session_control_batch4_seen_${TAG}_deployed.json
