#!/usr/bin/env python
"""Overlay what the policy predicts against what was demonstrated, per joint, over one episode.

Consumes the .npz written by `offline_chunk_eval.py --dump-traces`, so the curves come from
exactly the harness that produced the report numbers: same contamination filter, same
preprocessor, same padding mask.

The plot is a *fan*, not a single line, because that is what a chunk policy actually emits.
Each anchor produces one `chunk_size`-step open-loop prediction; drawing every one as its own
short segment from the frame it was made at shows three things a scalar MAE hides:

  * a constant offset  - the fan sits parallel to, but above/below, ground truth
  * horizon drift      - segments start on the black line and peel away from it
  * a null comparison  - `hold_state` is flat by construction, so any segment flatter than
                         ground truth is the policy approximating "do nothing"

    python plot_traces.py traces.npz --out traces.png
    python plot_traces.py traces.npz --episode 3 --joints 0 1 2 --out arm_left.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--episode", type=int, default=None, help="default: the first one in the dump")
    p.add_argument("--joints", type=int, nargs="*", default=None, help="default: all")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-null", action="store_true", help="hide the hold_state baseline")
    p.add_argument("--mode", choices=("fan", "executed"), default="fan",
                   help="fan: every anchor's whole chunk (default). executed: only the slice each "
                        "chunk actually contributes at n_action_steps, i.e. the single per-tick "
                        "curve the robot would follow -- what select_action returns.")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="executed mode only; default = chunk length (fully open loop)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    names = [str(n) for n in d["joint_names"]]
    ep = int(d["episode_index"][0]) if args.episode is None else args.episode
    sel = d["episode_index"] == ep
    if not sel.any():
        raise SystemExit(f"episode {ep} not in dump; present: {sorted(set(d['episode_index'].tolist()))}")

    pred, gt, state = d["pred"][sel], d["gt"][sel], d["state"][sel]
    valid, frame0 = d["valid"][sel], d["frame_index"][sel]
    joints = args.joints if args.joints else range(pred.shape[2])

    # In `executed` mode the policy is only re-queried every n_action_steps ticks, so most
    # anchors are never actually consulted on the robot. Keep the ones that would be, and
    # clip each to the slice it contributes; the concatenation is the per-tick action curve.
    n_exec = args.n_action_steps or pred.shape[1]
    if args.mode == "executed":
        step = max(1, round(n_exec / max(1, int(frame0[1] - frame0[0])))) if len(frame0) > 1 else 1
        keep = np.arange(0, len(pred), step)
        pred, gt, state, valid, frame0 = (x[keep] for x in (pred, gt, state, valid, frame0))
        valid = valid & (np.arange(valid.shape[1]) < n_exec)

    # Ground truth is reassembled from the anchors' own targets: anchor a covers frames
    # frame0[a] .. frame0[a]+H-1, and with stride < H those windows overlap and agree, so the
    # union is a dense recording of the episode with no second dataset read.
    fig, axes = plt.subplots(len(joints), 1, figsize=(13, 1.9 * len(joints)), sharex=True, squeeze=False)
    for ax, j in zip(axes[:, 0], joints):
        dense: dict[int, float] = {}
        for a in range(len(pred)):
            for h in np.flatnonzero(valid[a]):
                dense[int(frame0[a]) + int(h)] = gt[a, h, j]
        f = np.array(sorted(dense))
        ax.plot(f / args.fps, [dense[k] for k in f], color="black", lw=1.6, zorder=3,
                label="demonstrated" if j == joints[0] else None)

        for a in range(len(pred)):
            h = np.flatnonzero(valid[a])
            t = (frame0[a] + h) / args.fps
            ax.plot(t, pred[a, h, j], color="crimson",
                    lw=1.3 if args.mode == "executed" else 1.0,
                    alpha=0.9 if args.mode == "executed" else 0.55, zorder=2,
                    label=("executed action" if args.mode == "executed" else "predicted chunk")
                    if (a == 0 and j == joints[0]) else None)
            if not args.no_null:
                ax.plot(t, np.full(len(h), state[a, j]), color="tab:blue", lw=0.9, ls="--", alpha=0.4,
                        zorder=1, label="hold_state (null)" if (a == 0 and j == joints[0]) else None)

        ax.set_ylabel(names[j], fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("episode time (s)")
    fig.legend(loc="upper right", frameon=False, fontsize=9)
    what = ("per-tick executed action" if args.mode == "executed"
            else "open-loop chunk predictions")
    fig.suptitle(f"{str(d['repo_id'])} - episode {ep}: {what} vs. demonstration")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({len(pred)} anchors, episode {ep})")


if __name__ == "__main__":
    main()
