"""Verify the camera-embedding patch: load the PATCHED act module, leave the repo untouched."""

import importlib.machinery
import sys
import types

import torch

if "transformers" not in sys.modules:
    _stub = types.ModuleType("transformers")
    _stub.AutoProcessor = _stub.AutoTokenizer = _stub.PreTrainedTokenizerBase = object
    _stub.__version__ = "0.0.0-stub"
    _stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    _stub.__path__ = []
    sys.modules["transformers"] = _stub

SRC = "/home/kewei/YING/robot_data_platform/lerobot/src/lerobot"
PATCHED = "/tmp/claude-1000/-home-kewei-YING-paper/ed6a271b-9f98-40a6-b2f0-0affbdf2f743/scratchpad/patchwork/act"

# Everything resolves against the real repo except `lerobot.policies.act`, which points at the
# patched copies -- so this exercises the patch without writing to the repo.
for _name, _path in [("lerobot.policies", f"{SRC}/policies"), ("lerobot.policies.act", PATCHED)]:
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [_path]
    _pkg.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None, is_package=True)
    sys.modules[_name] = _pkg

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMS = ["top", "wrist_L", "wrist_R"]
H, W = 480, 640
PER_CAM = 15 * 20


def build(use_camera_embedding: bool):
    cfg = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            **{f"{OBS_IMAGES}.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)) for c in CAMS},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        pretrained_backbone_weights=None,
        use_camera_embedding=use_camera_embedding,
    )
    return cfg, ACT(cfg).eval()


def capture_pos(model):
    """Run a forward pass and return the positional embedding the encoder actually received."""
    grabbed = {}
    orig = type(model.encoder).forward

    def spy(self, x, pos_embed=None, key_padding_mask=None):
        grabbed.setdefault("pos", pos_embed)
        return orig(self, x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)

    type(model.encoder).forward = spy
    batch = {OBS_STATE: torch.randn(2, 16), OBS_IMAGES: [torch.rand(2, 3, H, W) for _ in CAMS]}
    with torch.no_grad():
        model(batch)
    type(model.encoder).forward = orig
    return grabbed["pos"]


for flag in (False, True):
    cfg, model = build(flag)
    pos = capture_pos(model)
    n_1d = pos.shape[0] - len(CAMS) * PER_CAM
    blocks = [pos[n_1d + i * PER_CAM : n_1d + (i + 1) * PER_CAM] for i in range(len(CAMS))]
    diffs = [(blocks[0] - blocks[i]).abs().max().item() for i in range(1, len(CAMS))]
    extra = sum(p.numel() for n, p in model.named_parameters() if "encoder_cam_id_embed" in n)
    print(f"use_camera_embedding={flag}")
    print(f"  cam0 vs cam1 max|diff| = {diffs[0]:.4f}   cam0 vs cam2 max|diff| = {diffs[1]:.4f}")
    print(f"  all camera pos blocks identical: {all(d == 0.0 for d in diffs)}")
    print(f"  added parameters: {extra}")

# --- gradients actually reach the new embedding -------------------------------
cfg, model = build(True)
model.train()
batch = {
    OBS_STATE: torch.randn(2, 16),
    OBS_IMAGES: [torch.rand(2, 3, H, W) for _ in CAMS],
    ACTION: torch.randn(2, cfg.chunk_size, 16),
    "action_is_pad": torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
}
actions, _ = model(batch)
actions.abs().mean().backward()
g = model.encoder_cam_id_embed.weight.grad
print(f"grad on encoder_cam_id_embed: norm={g.norm():.3e}, per-camera={[f'{r:.2e}' for r in g.norm(dim=1)]}")

# --- backward compatibility: an old checkpoint still loads ---------------------
_, old_model = build(False)
old_state = old_model.state_dict()
_, new_model = build(True)
missing, unexpected = new_model.load_state_dict(old_state, strict=False)
print(f"old checkpoint -> new model (strict=False): missing={missing}, unexpected={unexpected}")
print(f"old checkpoint -> old model (strict=True) : ", end="")
_, old_model2 = build(False)
old_model2.load_state_dict(old_state, strict=True)
print("OK")
