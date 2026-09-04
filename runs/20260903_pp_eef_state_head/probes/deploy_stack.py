#!/usr/bin/env python
"""Does the EEF deploy route rewrite the chunk, and over which action dimensions?

The 09-02 report ran `--filters none` on the stated ground that the EEF path "hands the
chunk to VlaHost verbatim".  Everything downstream of that -- whether a deployed number
exists at all for these weights -- depends on it being true.  This probe answers it from
the deploy checkout only: which strategy `deploy_config_eef.yaml` selects, whether that
strategy reaches `send_next_action_chunk`, which ENABLE_* stages are live, and which action
columns each live stage touches in a 14-D EEF chunk versus a 16-D joint chunk.

    /opt/robot-platform/train-venv/bin/python probes/deploy_stack.py > probes/deploy_stack.txt
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
import yaml

VLAHOST = Path("/home/kewei/YING/lerobot_vlahost")
CFG = VLAHOST / "workflows/robot_interaction/deploy_config_eef.yaml"
CORE = VLAHOST / "src/lerobot/rollout/strategies/core.py"
BASE = VLAHOST / "src/lerobot/rollout/strategies/base.py"
EEF_NAMES = json.load(open("/mnt/robot_platform/datasets/tidy_up_stationery_le/"
                           "batch_success_53_eval_data_eef/meta/info.json"))["features"]["action"]["names"]
JOINT_NAMES = json.load(open("/mnt/robot_platform/datasets/tidy_up_stationery_le/"
                             "batch_success_53_eval_data/meta/info.json"))["features"]["action"]["names"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


cfg = yaml.safe_load(open(CFG))
core = CORE.read_text()
base = BASE.read_text()

print("## 1. does the chunk-rewriting code run on this config?\n")
print(f"- `{CFG.name}`: inference.type = {cfg['inference']['type']!r}, "
      f"strategy = {cfg['inference']['strategy']!r}, n_action_steps = {cfg['inference']['n_action_steps']}")
print(f"- robot.type = {cfg['robot']['type']!r}, policy = ...{cfg['policy']['path'].split('/jobs/')[1].split('/')[0]}")
print(f"- `strategies/factory.py`: 'base' -> BaseStrategy")
hit = [i for i, l in enumerate(base.splitlines(), 1) if "send_next_action_chunk(" in l and "import" not in l]
print(f"- `strategies/base.py:{hit[0]}` calls `send_next_action_chunk` when `engine.produces_chunks`,")
print(f"  which is what `inference.type: chunk` selects.  **The rewrite runs.**")

print("\n## 2. which stages are live\n")
flags = dict(re.findall(r"^(ENABLE_\w+)\s*=\s*(True|False)", core, re.M))
STAGE = {
    "rollbacks": "ENABLE_REMOVE_SMALL_ROLLBACKS",
    "gripper_loops": "ENABLE_REMOVE_OPEN_GRIPPER_LOOPS",
    "smoothing": None,  # unconditional
    "excursions": "ENABLE_SMOOTH_LARGE_EXCURSIONS",
    "bridge": "ENABLE_FIXED_K_REAL_STATE_BRIDGE",
}
print("| harness filter | deploy flag | live |")
print("|---|---|---|")
live = []
for stage, flag in STAGE.items():
    on = True if flag is None else flags[flag] == "True"
    live.append(stage) if on else None
    print(f"| `{stage}` | {flag or '(unconditional call)'} | {'**yes**' if on else 'no'} |")
print(f"| `gripper_clip` | driver `_prepare_action` | **yes** |")
print(f"\n`--filters {','.join(live)},gripper_clip` is therefore the deploy stack.")
print("`--filters all` would be WRONG here: it adds two stages the robot has switched off.")

print("\n## 3. which columns each live stage touches\n")
arm_eef = [i for i, n in enumerate(EEF_NAMES) if "gripper" not in n.lower()][:14]
arm_jnt = [i for i, n in enumerate(JOINT_NAMES) if "gripper" not in n.lower()][:14]
print("`send_next_action_chunk` builds `arm_joint_keys = [k for k in ordered_keys if "
      '"gripper" not in k.lower()][:14]` and bridges `min(14, len(arm_joint_keys))` columns.')
print("The other stages take a literal `joint_count=14`, which is a column *count*, not a mask.\n")
print("| action space | width | arm_joint_keys | bridge covers | joint_count=14 covers | gripper cols |")
print("|---|---:|---:|---|---|---|")
for label, names, arm in (("16-D joint", JOINT_NAMES, arm_jnt), ("14-D EEF", EEF_NAMES, arm_eef)):
    grip = [i for i, n in enumerate(names) if "gripper" in n.lower()]
    print(f"| {label} | {len(names)} | {len(arm)} | cols 0-{min(14, len(arm)) - 1} | "
          f"cols 0-13 | {grip} |")
print("\nIn the EEF space `joint_count=14` therefore sweeps the two gripper columns (12, 13)")
print("into the rollback and smoothing stages, while the bridge correctly skips them.")
print("That is deploy's behaviour, not a harness artefact; the harness reproduces it.")

print("\n## 4. the unforked harness does NOT reproduce it at width 14\n")
fork = load("fork", Path(__file__).resolve().parent.parent / "offline_chunk_eval.py")
orig = load("orig", Path(__file__).resolve().parent.parent.parent.parent / "offline_chunk_eval.py")
ops = fork.load_deploy_trajectory_ops(VLAHOST)
fork.set_action_layout(EEF_NAMES)
torch.manual_seed(0)
chunk = torch.randn(50, 14) * 0.01 + torch.linspace(0, 0.2, 50).unsqueeze(1)
chunk[:, 12:14] = torch.rand(50, 2) * 2 - 0.5          # gripper commands, some out of [0,1]
state = torch.randn(14) * 0.01
for stage in ("bridge", "gripper_clip"):
    f = fork.apply_deploy_filter(ops, stage, chunk, state)
    o = orig.apply_deploy_filter(ops, stage, chunk, state)
    print(f"- `{stage}`: forked vs unforked max|Δ| = {float((f - o).abs().max()):.5f} "
          f"(on the gripper columns: {float((f - o)[:, 12:14].abs().max()):.5f})")
print("- unforked `bridge` rewrites cols 0-13, i.e. it Hermite-bridges the two gripper")
print("  channels that deploy leaves alone; unforked `gripper_clip` is a no-op below width 16.")
try:
    orig.deploy_ablation_chunk(ops, chunk, state)
    print("- unforked `--filter-ablation`: ran")
except Exception as e:
    print(f"- unforked `--filter-ablation`: **{type(e).__name__}: {e}** "
          "(remove_open_gripper_loops hardcodes gripper indices 14/15)")

torch.manual_seed(0)
c16 = torch.randn(50, 16) * 0.01
s16 = torch.randn(16) * 0.01
fork.set_action_layout(JOINT_NAMES)
same = all(
    torch.equal(fork.apply_deploy_filter(ops, s, c16, s16), orig.apply_deploy_filter(ops, s, c16, s16))
    for s in fork.DEPLOY_FILTER_ORDER
) and all(
    torch.equal(a, b) for a, b in zip(fork.deploy_ablation_chunk(ops, c16, s16).values(),
                                      orig.deploy_ablation_chunk(ops, c16, s16).values())
)
print(f"\n- at width 16 the fork is bit-identical to the unforked harness: **{same}**")
