#!/usr/bin/env bash
# ACT-DiT: the two 08-31 checkpoints nobody evaluated, on batch_success_53_eval_data.
#
# Reference arms are re-run here rather than quoted from the older reports: those numbers
# came from the 1st-gen harness in runs/scripts_act_eval_test/, and a cross-generation
# table is not one ruler.  Five checkpoints, one harness, one command shape.
#
# Two horizons per checkpoint, because the two prior reports use two different rulers:
#   h100 -> full chunk, comparable to act_dit-lowlr-diffusion §4
#   h50  -> executed window + deploy rewrite, comparable to act_dit-flowmatching-deployed §3
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400
PY=/opt/robot-platform/train-venv/bin/python      # trained these checkpoints; act_dit swaps
                                                  # the EMA shadow in on .eval() by itself
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
EVAL=$D/batch_success_53_eval_data
TRAIN=$D/batch_success_361                        # --train-root: action fingerprints, not dirname
HARNESS=../../offline_chunk_eval.py

$PY make_ckpt_shims.py
mkdir -p results

ACT_BASELINE=/mnt/robot_platform/jobs/act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328/run/checkpoints/200000/pretrained_model

run () {  # run <name> <checkpoint>
  for h in 100 50; do
    local out="results/eval53_$1_h$h.json"
    [ -s "$out" ] && { echo "skip $out"; continue; }
    echo "══ $1  horizon $h"
    $PY $HARNESS --checkpoint "$2" \
      --dataset-root "$EVAL" --train-root "$TRAIN" \
      --stride 20 --batch-size 32 --num-workers 8 \
      --n-action-steps "$h" $([ "$h" = 50 ] && echo --filter-ablation) \
      --out "$out"
  done
}

run fm_lowlr             ckpt/fm_lowlr              # 08-31_05-22 — the missing (fm, 1e-5, EMA) cell
run fm_lowlr_noadaln_rk4 ckpt/fm_lowlr_noadaln_rk4  # 08-31_06-17 — + state out of adaLN + rk4
run fm_hilr              ckpt/fm_hilr               # 08-27_04-32 — published reference
run diff_lowlr           ckpt/diff_lowlr            # 08-24_06-21 — published reference
run act_baseline         "$ACT_BASELINE"            # 08-17      — acceptance reference
