#!/usr/bin/env bash
# The EEF re-evaluation the 08-31 report asked for, on the two things that changed:
#
#   * batch_success_53_eval_data_eef -- the independent 53-episode eval set, now converted
#     to the 14-D EEF action space on the deploy machine.  Same 53 recordings as the joint
#     eval set (splits.txt proves it by observation.velocity fingerprint), so for the first
#     time the EEF and joint policy/null ratios are measured on the SAME sessions.
#   * act_eef ... batch_success_505_eef -- the ACT-EEF baseline retrained on the very
#     dataset patch_policy-EEF was trained on.  The 08-31 comparison had to use the weaker
#     361_eef baseline on 18 shared-unseen episodes.
#
# Both checkpoints carry bit-identical normaliser statistics (count 419330), so the
# snorlax-trained act_eef saw the same 505_eef data as the cluster-trained patch_policy.
#
# --filters none throughout: the EEF deploy path (marvain_m6_eef_ros_robot.send_action_chunk)
# hands the chunk to VlaHost verbatim, so policy_deployed == policy_raw by construction.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
PP=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints
A505=$J/act_eef_tidy_up_stationery_le_batch_success_505_eef/checkpoints
A361=$J/act_eef_tidy_up_stationery_le_batch_success_361_eef_2026-08-26_06-10-46-561630/run/checkpoints

run () { local name=$1; shift; echo "=== $name"; $PY offline_chunk_eval.py "$@" --out "$name.json" 2>&1 | tee "$name.log"; }

# Every run scores the same anchors: same eval root, same stride, and the train-root
# exclusion drops nothing (splits.txt: 0 of 53 episodes appear in any training set).
E=(--dataset-root "$D/batch_success_53_eval_data_eef"
   --train-root "$D/batch_success_505_eef" --train-root "$D/batch_success_361_eef"
   --stride 20 --batch-size 8 --num-workers 8 --filters none)

# horizon 50 = the executed window of deploy_config_patch_policy_eef.yaml
run pp_eef_200k       --checkpoint "$PP/200000/pretrained_model" "${E[@]}" --n-action-steps 50 --seed-repeat 1
run pp_eef_100k       --checkpoint "$PP/100000/pretrained_model" "${E[@]}" --n-action-steps 50

# matched baseline: same training set, same steps, same seed, chunk_size 100
run acteef_505_200k   --checkpoint "$A505/200000/pretrained_model" "${E[@]}" --n-action-steps 50
run acteef_505_100k   --checkpoint "$A505/100000/pretrained_model" "${E[@]}" --n-action-steps 50
# horizon 60 = the executed window of deploy_config_eef.yaml (ACT's chunk really is 100
# long, so unlike patch_policy's this is not silently truncated)
run acteef_505_h60    --checkpoint "$A505/200000/pretrained_model" "${E[@]}" --n-action-steps 60

# the 08-31 baseline, same eval set, to size how much of any change is just more data
run acteef_361_200k   --checkpoint "$A361/200000/pretrained_model" "${E[@]}" --n-action-steps 50



# The weight `deploy_config_eef.yaml` actually points at.  It was trained on
# batch_success_533_eef, which covers every episode of the 08-31 held-out split, so it had
# no scorable eval set until the independent 53-episode set was converted.
run acteef_533_200k --checkpoint "$J/act_eef_tidy_up_stationery_le_batch_success_533_eef_2026-08-27_14-48-42-880941/run/checkpoints/200000/pretrained_model" \
    --dataset-root "$D/batch_success_53_eval_data_eef" --train-root "$D/batch_success_533_eef" \
    --stride 20 --batch-size 8 --num-workers 8 --filters none --n-action-steps 50
echo "ALL DONE"
