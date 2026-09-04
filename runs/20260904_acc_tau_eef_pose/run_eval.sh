#!/usr/bin/env bash
set -uo pipefail
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
S=/tmp/claude-1000/-home-kewei-YING-paper/ba33f8c8-01a2-43b5-9a62-35695af7732a/scratchpad
cd /home/kewei/YING/paper/eval_policy
COMMON=(--dataset-root "$D/batch_success_53_eval_data_eef"
        --train-root "$D/batch_success_505_eef" --train-root "$D/batch_success_533_eef"
        --stride 20 --batch-size 8 --num-workers 8 --filters none --seed 0)
run () {  # name ckpt horizon
  echo "########## $1 (h$3)"
  $PY offline_chunk_eval.py --checkpoint "$2" "${COMMON[@]}" \
      --n-action-steps "$3" --out "$S/$1.json" 2>&1 | grep -v Warning | tail -18
}
run pp_eef_nostate $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-44-14-723349/run/checkpoints/200000/pretrained_model 60
run pp_eef_act5    $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-33-43-303120/run/checkpoints/200000/pretrained_model 60
run acteef_533_h60 $J/act_eef_tidy_up_stationery_le_batch_success_533_eef_2026-08-27_14-48-42-880941/run/checkpoints/200000/pretrained_model 60
run acteef_533_h50 $J/act_eef_tidy_up_stationery_le_batch_success_533_eef_2026-08-27_14-48-42-880941/run/checkpoints/200000/pretrained_model 50
echo "########## ALL DONE"
