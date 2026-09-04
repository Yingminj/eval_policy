#!/usr/bin/env python3
"""Shadow-checkpoint shims for act_dit weights trained before `state_in_adaln` existed.

Six of the seven act_dit jobs wrote `config.json` without the field, so they load against
the new default (False) and die on the adaLN Linear shape.  configuration_act_dit.py:56
says the fix is `"state_in_adaln": true`.  We never touch the archived checkpoint: each
shim is a directory of symlinks with one rewritten config.json.

The adaLN input width is the ground truth, not the config: 256 = timestep only
(state_in_adaln False), 256 + dim_model = state on adaLN (True).  We read it out of the
safetensors header and assert the config we write agrees.
"""
import json, os, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def adaln_in_features(safetensors: Path) -> int:
    with open(safetensors, "rb") as f:
        hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    key = "model.decoder.layers.0.adaln.1.weight"
    return hdr[key]["shape"][1]


def shim(src: Path, name: str) -> Path:
    cfg = json.load(open(src / "config.json"))
    want_state = adaln_in_features(src / "model.safetensors") > cfg["timestep_embed_dim"]
    expect = cfg["timestep_embed_dim"] + (cfg["dim_model"] if want_state else 0)
    got = adaln_in_features(src / "model.safetensors")
    assert got == expect, f"{name}: adaLN width {got} != {expect}"

    dst = HERE / "ckpt" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        if f.name != "config.json":
            link.symlink_to(f)
    cfg["state_in_adaln"] = want_state
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    print(f"{name:22s} state_in_adaln={str(want_state):5s} (adaLN in={got})  -> {dst}")
    return dst


def _demo():
    """Self-check: the width rule must round-trip both known cases."""
    for name, src in JOBS.items():
        cfg = json.load(open(Path(src) / "config.json"))
        w = adaln_in_features(Path(src) / "model.safetensors")
        assert w in (cfg["timestep_embed_dim"], cfg["timestep_embed_dim"] + cfg["dim_model"]), w
        if "state_in_adaln" in cfg:  # 06-17 declares it; the rule must agree with the file
            assert (w > cfg["timestep_embed_dim"]) == cfg["state_in_adaln"], name
    print("demo OK")


J = "/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_"
JOBS = {
    "fm_lowlr":       J + "2026-08-31_05-22-33-507248/run/checkpoints/200000/pretrained_model",
    "fm_lowlr_noadaln_rk4": J + "2026-08-31_06-17-39-059317/run/checkpoints/200000/pretrained_model",
    "fm_hilr":        J + "2026-08-27_04-32-02-437338/run/checkpoints/200000/pretrained_model",
    "diff_lowlr":     J + "2026-08-24_06-21-05-197422/run/checkpoints/200000/pretrained_model",
}

if __name__ == "__main__":
    _demo()
    for name, src in JOBS.items():
        shim(Path(src), name)
