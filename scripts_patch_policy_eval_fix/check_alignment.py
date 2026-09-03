#!/usr/bin/env python
"""The one thing the fork changes that can silently be wrong: history indexing.

`action_delta_indices` starts at 1 - n_obs_steps for patch_policy, so batch["action"]
carries n_obs_steps - 1 rows of *past* actions before the chunk the newest observation is
responsible for. Off by those rows and every number in the report is a lagged comparison
that still looks plausible. Same for observation.state, where the anchor -- the pose the
deploy Hermite bridge starts from -- is the LAST frame of the window, not the first.

Checks both against the same dataset loaded without any delta_timestamps.

    /opt/robot-platform/train-venv/bin/python check_alignment.py
"""
import sys
from pathlib import Path

import torch

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361"
    "_2026-08-30_11-27-35-272413/run/checkpoints/200000/pretrained_model"
)
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data"
)
REPO = f"{ROOT.parent.name}/{ROOT.name}"

import lerobot.policies.factory  # noqa: E402,F401  registers the config subclasses
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

cfg = PreTrainedConfig.from_pretrained(CKPT)
meta = LeRobotDatasetMetadata(repo_id=REPO, root=ROOT)
dt = resolve_delta_timestamps(cfg, meta)
n_obs = getattr(cfg, "n_obs_steps", 1)
a_off = -min(list(cfg.action_delta_indices or [0]) + [0])
print(f"{cfg.type}: n_obs_steps={n_obs}, action window {cfg.action_delta_indices[0]}"
      f"..{cfg.action_delta_indices[-1]}, computed action offset {a_off}")
assert a_off == n_obs - 1, (a_off, n_obs)

ds_hist = LeRobotDataset(REPO, root=ROOT, delta_timestamps=dt)
ds_flat = LeRobotDataset(REPO, root=ROOT)

# Anchors far enough inside an episode that no window is clamped at an edge.
for i in (5000, 12345, 20000, 33333):
    h, f = ds_hist[i], ds_flat[i]
    assert h["observation.state"].shape[0] == n_obs, h["observation.state"].shape
    # newest frame of the observation window is this frame
    assert torch.equal(h["observation.state"][-1], f["observation.state"]), i
    # ... and the frame BEFORE it is not, or the window is degenerate and the whole
    # history argument is moot
    if n_obs > 1:
        assert not torch.equal(h["observation.state"][0], f["observation.state"]), i
    # delta 0 of the action window is this frame's own action
    assert torch.equal(h["action"][a_off], f["action"]), i
    # scoring from index 0 instead would compare against an action n_obs-1 ticks in the past
    if n_obs > 1:
        assert not torch.equal(h["action"][0], f["action"]), i
    # the chunk that gets scored must be the future, not the past
    assert h["action"].shape[0] - a_off == cfg.action_chunk_size, h["action"].shape
print("alignment OK: observation anchor = newest frame, action offset = n_obs_steps - 1")
