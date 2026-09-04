#!/usr/bin/env bash
ulimit -n "$(ulimit -Hn)"; export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
D=/mnt/robot_platform/datasets/tidy_up_stationery_le; J=/mnt/robot_platform/jobs
S=/tmp/claude-1000/-home-kewei-YING-paper/ba33f8c8-01a2-43b5-9a62-35695af7732a/scratchpad
cd /home/kewei/YING/paper/eval_policy
for n in pp_eef_state:2026-08-31_13-14-33-857400 pp_eef_nostate:2026-09-03_09-44-14-723349; do
  name=${n%%:*}; job=${n##*:}
  /opt/robot-platform/train-venv/bin/python -u offline_chunk_eval.py \
    --checkpoint $J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_$job/run/checkpoints/200000/pretrained_model \
    --dataset-root $D/batch_success_53_eval_data_eef \
    --train-root $D/batch_success_505_eef --train-root $D/batch_success_533_eef \
    --stride 20 --batch-size 8 --num-workers 8 --filters none --seed 1 \
    --n-action-steps 60 --out $S/${name}_seed1.json > $S/${name}_seed1.log 2>&1
  echo "$name seed1 exit=$?" >> $S/noise4.out
done
echo "DONE" >> $S/noise4.out
