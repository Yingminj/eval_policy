#!/usr/bin/env bash
# The two 09-03 patch_policy-EEF weights, against the 08-31 weight they were derived from
# and against the weight deploy_config_eef.yaml actually points at.
#
# What changed, read out of config.json rather than job.json (job.json is identical for all
# three: 200k steps / bs 16 / seed 1000 / lr 5.5e-5 constant):
#
#   pp_eef_state    08-31_13-14-33  diffusion  n_obs 2  use_robot_state TRUE   <- the baseline
#   pp_eef_nostate  09-03_09-44-14  diffusion  n_obs 2  use_robot_state FALSE  <- ONE variable
#   pp_eef_act5     09-03_09-33-43  act        n_obs 5  use_robot_state FALSE  <- THREE variables
#
#   (09-33 also moves gpt_block_size 2->25 and n_vqvae_training_steps 5000->20000; both are
#    read only under action_head="vqbet" -- see probes/config_diff.txt -- so they are inert.)
#
# Horizon: --n-action-steps 60 from deploy_config_eef.yaml (inference.n_action_steps).  The
# harness clamps it to the policy's own chunk, so patch_policy is scored over 50 steps (its
# action_chunk_size) and act_eef over 60 (chunk_size 100).  That clamp is not a harness
# artefact: patch_policy physically cannot fill the 60-step window this config asks for.
#
# Filters: the EEF route DOES have a deploy filter stack.  deploy_config_eef.yaml sets
# strategy=base + inference.type=chunk, so BaseStrategy.run calls send_next_action_chunk,
# which rewrites the chunk.  Per strategies/core.py:52-55 the live stack is
# rollbacks + smoothing + K=40 bridge (gripper_loops and excursions are ENABLE_*=False),
# plus the driver's gripper clip.  That is what --filters passes here.  The 09-02 report ran
# --filters none on the claim that the EEF path "hands the chunk to VlaHost verbatim"; that
# claim is contradicted by the code (probes/deploy_stack.txt).
#
# --filter-ablation additionally scores each stage cumulatively, sharing one rewrite.
set -euo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
HARNESS=./offline_chunk_eval.py          # forked; see fork.diff and README "harness fork"

PP_STATE=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400/run/checkpoints
PP_NOSTATE=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-44-14-723349/run/checkpoints
PP_ACT5=$J/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-33-43-303120/run/checkpoints
ACTEEF533=$J/act_eef_tidy_up_stationery_le_batch_success_533_eef_2026-08-27_14-48-42-880941/run/checkpoints

MAX="${MAX_ANCHORS:-}"                   # MAX_ANCHORS=50 ./run_eval.sh  -> smoke pass
SUF="${RESULT_SUFFIX:-}"                 # smoke results go to results/smoke_*
mkdir -p results

# Every run scores the SAME anchors: same eval root, same stride, and both training sets
# are excluded by action fingerprint in every run so the exclusion cannot differ between
# them (splits.txt: 0 of 53 episodes appear in either, so nothing is actually dropped).
# --batch-size is identical everywhere on purpose: the diffusion seed is re-set per batch,
# so two runs are only comparable at the same batch size.
COMMON=(--dataset-root "$D/batch_success_53_eval_data_eef"
        --train-root "$D/batch_success_505_eef"
        --train-root "$D/batch_success_533_eef"
        --stride 20 --batch-size 8 --num-workers 8
        --filters rollbacks,smoothing,bridge,gripper_clip --filter-ablation
        --seed 0)
[ -n "$MAX" ] && COMMON+=(--max-anchors-per-dataset "$MAX")

run () {  # run <name> <checkpoint> [extra flags...]
  local name="$1$SUF"; local ckpt="$2"; shift 2
  local out="results/$name.json"
  [ -s "$out" ] && { echo "skip $out"; return; }
  echo "══════ $name"
  $PY "$HARNESS" --checkpoint "$ckpt" "${COMMON[@]}" "$@" --out "$out" 2>&1 | tee "results/$name.log"
}

# --seed-repeat 1 on all three patch_policy weights.  For the two diffusion heads it sizes
# the sampling noise; for the act head it is the control -- a deterministic head must land
# on exactly the same number, and if it does not, the head is not what config.json says.
run pp_eef_state_h60    "$PP_STATE/200000/pretrained_model"   --n-action-steps 60 --seed-repeat 1
run pp_eef_nostate_h60  "$PP_NOSTATE/200000/pretrained_model" --n-action-steps 60 --seed-repeat 1
run pp_eef_act5_h60     "$PP_ACT5/200000/pretrained_model"    --n-action-steps 60 --seed-repeat 1

# Half-way weights for the two new arms only: whether either is still improving at 200k is
# the difference between "this configuration is worse" and "this configuration needs longer".
run pp_eef_nostate_h60_100k "$PP_NOSTATE/100000/pretrained_model" --n-action-steps 60
run pp_eef_act5_h60_100k    "$PP_ACT5/100000/pretrained_model"    --n-action-steps 60

# The weight deploy_config_eef.yaml points at, on the same ruler.  h60 is its real executed
# window; h50 exists only so policy_raw is directly comparable to the 09-02 report's table.
run acteef_533_h60 "$ACTEEF533/200000/pretrained_model" --n-action-steps 60
run acteef_533_h50 "$ACTEEF533/200000/pretrained_model" --n-action-steps 50

echo "ALL DONE"
