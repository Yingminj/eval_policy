#!/usr/bin/env bash
# The two 2026-08-31 patch_policy weights, each on the only eval set its action space has.
#
#   vqbet_*   16-D joint space, trained on batch_success_361, scored on the independent
#             53-episode eval set -- the same anchors as scripts_patch_policy_eval_fix.
#   eef_*     14-D EEF space, trained on batch_success_505_eef.  No EEF version of the
#             53-episode eval set exists (it was never converted from the bags), so the
#             held-out set is the 67 episodes of batch_success_533_eef whose action array
#             is absent from batch_success_505_eef -- a random split, not an independent
#             session.  Numbers are NOT comparable across the two blocks.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
VQ=$J/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-19-40-151329/run/checkpoints
EEF=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints
ACTEEF=$J/act_eef_tidy_up_stationery_le_batch_success_361_eef_2026-08-26_06-10-46-561630/run/checkpoints/last/pretrained_model

run () { local name=$1; shift; echo "=== $name"; $PY offline_chunk_eval.py "$@" --out "$name.json" 2>&1 | tee "$name.log"; }

# ---- 16-D joint space: same harness, same anchors, same filter ladder as the 08-30 report
JOINT=(--dataset-root "$D/batch_success_53_eval_data" --train-root "$D/batch_success_361"
       --stride 20 --batch-size 8 --num-workers 8 --filter-ablation)
run vqbet_100k      --checkpoint "$VQ/100000/pretrained_model" "${JOINT[@]}" --n-action-steps 50 --seed-repeat 1
run vqbet_100k_h8   --checkpoint "$VQ/100000/pretrained_model" "${JOINT[@]}" --n-action-steps 8
run vqbet_050k      --checkpoint "$VQ/050000/pretrained_model" "${JOINT[@]}" --n-action-steps 50

# ---- 14-D EEF space.  The joint filter ladder is meaningless here (it indexes arm joints
# 0..13 and grippers 14..15), and the EEF deploy path sends the chunk verbatim to VlaHost,
# so --filters none is the deployed chunk.
EEFA=(--dataset-root "$D/batch_success_533_eef" --train-root "$D/batch_success_505_eef"
      --stride 20 --batch-size 8 --num-workers 8 --filters none)
run eef_200k        --checkpoint "$EEF/200000/pretrained_model" "${EEFA[@]}" --n-action-steps 50 --seed-repeat 1
run eef_200k_h60    --checkpoint "$EEF/200000/pretrained_model" "${EEFA[@]}" --n-action-steps 60
run eef_100k        --checkpoint "$EEF/100000/pretrained_model" "${EEFA[@]}" --n-action-steps 50

# ---- head-to-head against ACT-EEF.  act_eef was trained on batch_success_361_eef, which
# covers 49 of those 67 episodes, so the fair set is the 18 episodes unseen by BOTH.
FAIR=(--dataset-root "$D/batch_success_533_eef"
      --train-root "$D/batch_success_505_eef" --train-root "$D/batch_success_361_eef"
      --stride 20 --batch-size 8 --num-workers 8 --filters none)
run eef_200k_fair       --checkpoint "$EEF/200000/pretrained_model" "${FAIR[@]}" --n-action-steps 50
run eef_200k_fair_h60   --checkpoint "$EEF/200000/pretrained_model" "${FAIR[@]}" --n-action-steps 60
run acteef_361_fair     --checkpoint "$ACTEEF" "${FAIR[@]}" --n-action-steps 50
run acteef_361_fair_h60 --checkpoint "$ACTEEF" "${FAIR[@]}" --n-action-steps 60

echo "ALL DONE"
