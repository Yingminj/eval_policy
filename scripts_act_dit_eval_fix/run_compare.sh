#!/usr/bin/env bash
# Comparators under the same deployment-faithful harness:
#   act   = ACT baseline, the checkpoint the fix-harness README reports
#   fm0820 = the earlier act_dit flow-matching+EMA run whose config is byte-identical to 08-27
set -euo pipefail
O=/home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_act_dit_eval_fix
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
S=/home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_act_eval_test_fix
PY=/opt/robot-platform/train-venv/bin/python
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
cd "$S"

while pgrep -f "offline_chunk_eval.py --checkpoint" >/dev/null; do sleep 10; done

# 1. predecessor act_dit run (same config, trained 6 days earlier) -- eval53 + in-training
FM=/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-20_22-59-31-790981/run/checkpoints/last/pretrained_model
for pair in "eval53:$D/batch_success_53_eval_data" "intrain:$D/batch_success_361"; do
  tag=${pair%%:*}; root=${pair#*:}
  echo "### fm0820 $tag"
  $PY offline_chunk_eval.py --checkpoint "$FM" --stride 20 --batch-size 32 --num-workers 8 \
    --n-action-steps 50 --filter-ablation --dataset-root "$root" --out $O/${tag}_fm0820_deployed.json
done

# 2. ACT baseline on the designated eval set (the fix README covers the other three conditions)
ACT=/mnt/robot_platform/jobs/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/last/pretrained_model
echo "### act eval53"
$PY offline_chunk_eval.py --checkpoint "$ACT" --stride 20 --batch-size 32 --num-workers 8 \
  --n-action-steps 50 --filter-ablation --train-root $D/batch_success_361 \
  --dataset-root $D/batch_success_53_eval_data --out $O/eval53_act_deployed.json
