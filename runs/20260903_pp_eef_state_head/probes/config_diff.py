#!/usr/bin/env python
"""What actually differs between the four checkpoints -- and which of those fields the code reads.

job.json is identical for all three patch_policy runs (200k / bs 16 / seed 1000), so "the
parameter changes" live entirely in the policy `config.json`.  Two of the fields that move
(`gpt_block_size`, `n_vqvae_training_steps`) are vqbet-only; this probe does not assert that
from the docstring, it greps every read site in the installed policy source and reports the
guard each one sits behind, so the report can say "inert" with a line number.

    /opt/robot-platform/train-venv/bin/python probes/config_diff.py > probes/config_diff.txt
"""
import json
import re
from pathlib import Path

J = Path("/mnt/robot_platform/jobs")
CKPT = {
    "pp_eef_state": J / "patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400",
    "pp_eef_nostate": J / "patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-44-14-723349",
    "pp_eef_act5": J / "patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-09-03_09-33-43-303120",
    "acteef_533": J / "act_eef_tidy_up_stationery_le_batch_success_533_eef_2026-08-27_14-48-42-880941",
}
SRC = Path("/opt/robot-platform/train-venv/lib/python3.12/site-packages/lerobot/policies/patch_policy")

cfg = {n: json.load(open(p / "run/checkpoints/200000/pretrained_model/config.json")) for n, p in CKPT.items()}
job = {n: json.load(open(p / "job.json")) for n, p in CKPT.items() if (p / "job.json").is_file()}

print("## 1. job-level training config\n")
FIELDS = ("dataset_repo_id", "steps", "batch_size", "seed", "num_workers", "save_freq", "resume")
print("| field | " + " | ".join(job) + " |")
print("|---|" + "---|" * len(job))
for f in FIELDS:
    print(f"| `{f}` | " + " | ".join(str(job[n]["config"].get(f)) for n in job) + " |")
for f in ("current_loss", "grad_norm", "current_lr"):
    print(f"| final `{f}` | " + " | ".join(str(job[n]["metrics"].get(f)) for n in job) + " |")
print(f"| wall-clock (s) | " + " | ".join(f"{job[n]['ended_at'] - job[n]['started_at']:.0f}" for n in job) + " |")
print(f"| node | " + " | ".join(str(job[n].get("node_name")) for n in job) + " |")

pp = [n for n in cfg if cfg[n]["type"] == "patch_policy"]
print("\n## 2. policy config.json: fields that differ across the three patch_policy weights\n")
keys = sorted(set().union(*(set(cfg[n]) for n in pp)))
diff = [k for k in keys if len({json.dumps(cfg[n].get(k), sort_keys=True) for n in pp}) > 1]
print("| field | " + " | ".join(pp) + " |")
print("|---|" + "---|" * len(pp))
for k in diff:
    print(f"| `{k}` | " + " | ".join(str(cfg[n].get(k)) for n in pp) + " |")
print(f"\nfields compared: {len(keys)}, differing: {len(diff)}")

print("\n## 3. fields that are equal across all three (the controls)\n")
SAME = ("action_chunk_size", "n_action_steps", "vision_encoder", "freeze_vision_encoder",
        "resize_shape", "dim_model", "optimizer_lr", "num_inference_steps", "beta_schedule",
        "normalization_mapping")
for k in SAME:
    vals = {json.dumps(cfg[n].get(k), sort_keys=True) for n in pp}
    mark = "same" if len(vals) == 1 else "DIFFERS"
    print(f"- `{k}` = {cfg[pp[0]].get(k)}  [{mark}]")

print("\n## 4. where the moved fields are actually read\n")
src = {p.name: p.read_text().splitlines() for p in sorted(SRC.glob("*.py"))}
for field in ("use_robot_state", "n_obs_steps", "action_head", "gpt_block_size", "n_vqvae_training_steps"):
    print(f"\n### `{field}`")
    hits = 0
    for fname, lines in src.items():
        for i, line in enumerate(lines, 1):
            if not re.search(rf"\b{field}\b", line) or line.lstrip().startswith("#"):
                continue
            if fname == "configuration_patch_policy.py" and i < 300:
                continue  # the dataclass declaration and its docstring, not a read site
            # nearest enclosing `if ... action_head ...` guard above this line
            guard = ""
            for j in range(i - 1, max(0, i - 40), -1):
                m = re.search(r"action_head\s*==\s*[\"'](\w+)[\"']", lines[j - 1])
                if m and len(lines[j - 1]) - len(lines[j - 1].lstrip()) < len(line) - len(line.lstrip()):
                    guard = f"  <- under `action_head == \"{m.group(1)}\"`"
                    break
            if not guard:  # otherwise, name the class it lives in -- some are vqbet-only wholesale
                for j in range(i - 1, 0, -1):
                    m = re.match(r"class (\w+)", lines[j - 1])
                    if m:
                        guard = f"  <- in class `{m.group(1)}`"
                        break
            hits += 1
            print(f"  {fname}:{i}: {line.strip()[:95]}{guard}")
    if not hits:
        print("  (no read site outside the dataclass declaration)")

print("\n## 5. acteef_533, for the deploy comparison\n")
c = cfg["acteef_533"]
print(f"- type={c['type']}, n_obs_steps={c.get('n_obs_steps')}, chunk_size={c.get('chunk_size')}, "
      f"n_action_steps={c.get('n_action_steps')}")
print(f"- deploy_config_eef.yaml asks for inference.n_action_steps = 60")
for n in pp:
    print(f"- {n}: action_chunk_size={cfg[n]['action_chunk_size']} -> the 60-step window is "
          f"{'SHORT BY ' + str(60 - cfg[n]['action_chunk_size']) + ' steps' if cfg[n]['action_chunk_size'] < 60 else 'filled'}")
