"""Confirm that LeRobot ACT gives all cameras byte-identical position embeddings."""

import importlib.machinery
import sys
import types

import torch

# The ambient env has a transformers/huggingface-hub version clash that trips an import guard in
# lerobot.processor.tokenizer_processor. ACT never touches the tokenizer, so stub it out.
if "transformers" not in sys.modules:
    _stub = types.ModuleType("transformers")
    _stub.AutoProcessor = _stub.AutoTokenizer = object
    _stub.PreTrainedTokenizerBase = object
    _stub.__version__ = "0.0.0-stub"
    _stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    _stub.__path__ = []
    sys.modules["transformers"] = _stub

SRC = "/home/kewei/YING/robot_data_platform/lerobot/src/lerobot"

# `lerobot.policies.__init__` eagerly imports every policy (EO1 needs the real transformers), and
# `lerobot.policies.act.__init__` pulls in the tokenizer processor. Register namespace stubs so the
# submodules resolve by path without either __init__ running.
for _name, _path in [("lerobot.policies", f"{SRC}/policies"), ("lerobot.policies.act", f"{SRC}/policies/act")]:
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

cfg = ACTConfig(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        **{f"{OBS_IMAGES}.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, H, W)) for c in CAMS},
    },
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    pretrained_backbone_weights=None,
)
model = ACT(cfg).eval()

# ---- 1. the position embedding never reads the feature values ----------------
fmap_a = torch.randn(1, 512, 15, 20)
fmap_b = torch.randn(1, 512, 15, 20) * 100 + 7  # wildly different content, same shape
pe_a = model.encoder_cam_feat_pos_embed(fmap_a)
pe_b = model.encoder_cam_feat_pos_embed(fmap_b)
print(f"[1] pos_embed(feat_a) == pos_embed(feat_b): {torch.equal(pe_a, pe_b)}  "
      f"(max|diff| = {(pe_a - pe_b).abs().max():.3e})")

# ---- 2. end-to-end: capture what the encoder actually receives ----------------
captured = {}
orig_forward = type(model.encoder).forward


def spy(self, x, pos_embed=None, key_padding_mask=None):
    captured.setdefault("pos", pos_embed)
    captured.setdefault("tok", x)
    return orig_forward(self, x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)


type(model.encoder).forward = spy
batch = {
    OBS_STATE: torch.randn(2, 16),
    OBS_IMAGES: [torch.rand(2, 3, H, W) for _ in CAMS],
}
with torch.no_grad():
    model(batch)
type(model.encoder).forward = orig_forward

pos = captured["pos"]  # (S, 1, D)
tok = captured["tok"]  # (S, B, D)
n_1d = pos.shape[0] - len(CAMS) * 15 * 20
per_cam = 15 * 20
print(f"[2] encoder input: {tok.shape[0]} tokens = {n_1d} 1-D + {len(CAMS)}x{per_cam} image")

blocks = [pos[n_1d + i * per_cam : n_1d + (i + 1) * per_cam] for i in range(len(CAMS))]
for i in range(1, len(CAMS)):
    same = torch.equal(blocks[0], blocks[i])
    print(f"    pos_embed block cam0({CAMS[0]}) == cam{i}({CAMS[i]}): {same}"
          f"   max|diff| = {(blocks[0] - blocks[i]).abs().max():.3e}")

# ---- 3. does anything else disambiguate the cameras? --------------------------
print(f"[3] shared backbone across cameras: {not isinstance(model.backbone, (list, torch.nn.ModuleList))}"
      f"  (type={type(model.backbone).__name__})")
print(f"    encoder_1d_feature_pos_embed covers only {model.encoder_1d_feature_pos_embed.num_embeddings}"
      f" non-image tokens")
n_cam_specific = sum(
    p.numel() for n, p in model.named_parameters() if "cam" in n.lower() and "pos_embed" not in n
)
print(f"    parameters whose name mentions a camera identity: {n_cam_specific}")
