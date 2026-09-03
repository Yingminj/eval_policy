#!/usr/bin/env python
"""Plot per-horizon action error from offline_chunk_eval.py report JSONs.

    python plot_horizon.py --out fig.png \
        --curve "in-training"  intrain_control.json  policy \
        --curve "held-out"     heldout_clean.json    policy \
        --curve "hold_state (held-out)" heldout_clean.json hold_state
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curve", nargs=3, action="append", metavar=("LABEL", "JSON", "KEY"), default=[])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--title", default="Open-loop action-chunk error vs. prediction horizon")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = [("-", "#1f77b4"), ("-", "#d62728"), ("--", "#7f7f7f"), ("--", "#2ca02c"), (":", "#9467bd")]

    for i, (label, path, key) in enumerate(args.curve):
        y = json.loads(Path(path).read_text())["aggregate"][key]["mae_per_horizon"]
        x = [h / args.fps for h in range(len(y))]
        ls, c = styles[i % len(styles)]
        ax.plot(x, y, ls, color=c, lw=1.9, label=label)

    ax.set_xlabel("prediction horizon (seconds into the chunk)")
    ax.set_ylabel("mean absolute action error (joint units)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
