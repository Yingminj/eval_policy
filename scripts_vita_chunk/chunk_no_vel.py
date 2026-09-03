"""Drive the real VITA checkpoint through ChunkInferenceEngine + marvain_m6_http payload build."""
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.rollout.inference.chunk import ChunkInferenceEngine
from lerobot.utils.constants import ACTION

CKPT = "/mnt/robot_platform/jobs/vita_tidy_up_stationery_le_batch_2_2026-08-12_02-57-02-431745/run/checkpoints/last/pretrained_model"

JOINTS = [f"left_arm_joint_{i}" for i in range(1, 8)] + [
    f"right_arm_joint_{i}" for i in range(1, 8)
] + ["left_gripper", "right_gripper"]
ACTION_KEYS = [f"{j}.pos" for j in JOINTS]

cfg = PreTrainedConfig.from_pretrained(CKPT, cli_overrides=["--device=cpu"])
cfg.pretrained_path = CKPT
policy = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg).to("cpu").eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=CKPT,
    preprocessor_overrides={
        "device_processor": {"device": "cpu"},
        "rename_observations_processor": {"rename_map": {}},
    },
)

dataset_features = {ACTION: {"dtype": "float32", "shape": (16,), "names": ACTION_KEYS}}

engine = ChunkInferenceEngine(
    policy=policy,
    preprocessor=pre,
    postprocessor=post,
    dataset_features=dataset_features,
    ordered_action_keys=ACTION_KEYS,
    task="tidy up stationery",
    device="cpu",
    robot_type="marvain_m6_http",
    n_action_steps=80,          # what deploy_config_chunk.yaml asks for
    chunk_interval_s=0.0,
)
engine.reset()


def fake_frame():
    """What the rollout loop hands the engine: raw HWC uint8 images + float32 state."""
    frame = {
        "observation.state": np.random.randn(16).astype(np.float32),

    }
    for cam in ("top", "wrist_L", "wrist_R"):
        frame[f"observation.images.{cam}"] = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
    return frame


chunk = engine.get_action_chunk(fake_frame())
print("chunk shape:", tuple(chunk.shape), "dtype:", chunk.dtype)
assert chunk.ndim == 2 and chunk.shape[1] == 16, chunk.shape

# --- feed it to the robot payload builder, no HTTP ---
from lerobot.robots.marvain_m6_http import MarvainM6HttpRobot, MarvainM6HttpRobotConfig

robot = MarvainM6HttpRobot(MarvainM6HttpRobotConfig(cameras={}))
actions = [dict(zip(ACTION_KEYS, row.tolist())) for row in chunk]
payload, result = robot._prepare_action(actions[0])
print("payload keys:", sorted(payload))
print("jointcmd_left :", [round(v, 4) for v in payload["jointcmd_left"]])
print("jointcmd_right:", [round(v, 4) for v in payload["jointcmd_right"]])
print("gripper_left/right:", round(payload["gripper_left"], 4), round(payload["gripper_right"], 4))
assert len(payload["jointcmd_left"]) == 7 and len(payload["jointcmd_right"]) == 7
print("OK")
