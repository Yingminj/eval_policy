#!/usr/bin/env bash
# acc@tau + 末端位姿误差：四个 09-03 权重，同一批 anchor,只换尺子。
#
# 与 20260903_pp_eef_state_head/run_eval.sh 的唯一差别是 --filters none。
# 那轮跑 rollbacks,smoothing,bridge,gripper_clip --filter-ablation,靠的是 run 目录里
# 的 harness 分叉修好了 16-D 关节硬编码;顶层 harness 至今没有那个修复,在 14-D EEF 上
# 桥会重写夹爪列、clip 会变成空操作、--filter-ablation 直接崩。所以本轮没有部署列。
#
# --stride / --batch-size / --seed / 两个 --train-root 与那轮逐字相同,anchor 集合因此
# 完全一致(2007 个,丢弃 0),两份报告的数字可以直接比。
set -uo pipefail
cd "$(dirname "$0")"
ulimit -n "$(ulimit -Hn)"
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400

PY=/opt/robot-platform/train-venv/bin/python
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
J=/mnt/robot_platform/jobs
HARNESS=../../offline_chunk_eval.py       # 顶层,未分叉;改动见 harness.diff
mkdir -p results

COMMON=(--dataset-root "$D/batch_success_53_eval_data_eef"
        --train-root "$D/batch_success_505_eef"
        --train-root "$D/batch_success_533_eef"
        --stride 20 --batch-size 8 --num-workers 8
        --filters none --seed 0)

PP=patch_policy_tidy_up_stationery_le_batch_success_505_eef
AE=act_eef_tidy_up_stationery_le_batch_success_533_eef

run () {  # 短名 job 目录 horizon
  echo "########## $1 (h$3)"
  $PY "$HARNESS" --checkpoint "$J/$2/run/checkpoints/200000/pretrained_model" \
      "${COMMON[@]}" --n-action-steps "$3" --out "results/$1.json"
}

# 串行。并发跑会几个进程抢同一张 4090,慢到看起来像挂了。
run pp_eef_state   ${PP}_2026-08-31_13-14-33-857400 60   # chunk 50 -> 实际 h50
run pp_eef_nostate ${PP}_2026-09-03_09-44-14-723349 60   # chunk 50 -> 实际 h50
run pp_eef_act5    ${PP}_2026-09-03_09-33-43-303120 60   # chunk 50 -> 实际 h50
run acteef_533_h60 ${AE}_2026-08-27_14-48-42-880941 60   # chunk 100 -> 实际 h60
run acteef_533_h50 ${AE}_2026-08-27_14-48-42-880941 50   # 与三个 patch_policy 同尺
echo "########## ALL DONE"
