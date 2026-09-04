#!/usr/bin/env python
"""Deployment-faithful offline action-chunk evaluation of a trained LeRobot checkpoint.

Fork of ``scripts_act_eval_test_fix/offline_chunk_eval.py`` with observation-history
support, so ``patch_policy`` (n_obs_steps > 1) can be scored on the same anchors, with the
same deploy rewrite and the same null baselines as the ACT checkpoints.  Three differences,
all forced by the policy rather than chosen:

  * ``observation.*`` arrives as ``(B, n_obs_steps, ...)``; the anchor is the *last* frame,
    which is the one the deploy bridge and ``hold_state`` must be anchored on.
  * ``action`` arrives as ``(B, n_obs_steps - 1 + action_chunk_size, A)`` because
    ``action_delta_indices`` starts negative; delta 0 sits at index ``n_obs_steps - 1``.
    Scoring from index 0 would compare the chunk against the *past*.
  * ``PatchPolicy.predict_action_chunk`` reads its internal deployment queues, not the
    batch, so the batched path calls ``policy.model.predict`` with the images stacked the
    way ``select_action`` stacks them.  The diffusion head samples noise, so ``--seed``
    fixes it and ``--seed-repeat`` measures what the sampling alone is worth.

What this is
------------
A fork of ``scripts_act_eval_test/offline_chunk_eval.py`` that scores *the action
sequence the robot is actually commanded to follow*, not the raw tensor the policy
emits.  On this cluster those are not the same thing.  Between
``predict_action_chunk`` and the joints there is:

  1. ``inference.n_action_steps`` (50) — only the first half of the 100-step chunk is
     ever dispatched (``deploy_config_act_dit.yaml``).
  2. ``lerobot/rollout/strategies/core.py:send_next_action_chunk`` — the chunk is
     rewritten before it is sent: small rollbacks removed, open-gripper loops removed,
     one binomial smoothing pass over the 14 arm joints, large excursions linearised,
     and finally a **fixed K=40 cubic-Hermite bridge** that replaces the first 40 of the
     50 arm-joint steps with a zero-start-velocity S-curve from the *measured* joint
     state to step 40 of the smoothed chunk.  Only ~10 of the 50 executed steps are
     unmodified policy output.
  3. ``marvain_m6_http._prepare_action`` — gripper targets clipped to [0, 1].

Scoring the raw chunk therefore answers a question nobody asks on the robot.  This
harness reports both, on the same anchors and the same padding mask:

  * ``policy_raw``      — the chunk as the policy emits it, truncated to the executed
                          horizon.  Comparable to the numbers in the original report.
  * ``policy_deployed`` — the same chunk after the full deploy rewrite above.  This is
                          what ``/action_chunk`` receives and what the 500 Hz player
                          linearly interpolates onto the arms.
  * ``hold_state`` / ``train_mean`` — unchanged null baselines.

Metrics
-------
Every variant is reduced the same way, and the reduction set is fixed rather than
selectable: two runs of the same checkpoint must produce diffable reports, and a metric
that appears only when someone thought to ask for it cannot be compared against a run
where nobody did.  Alongside the MAE/RMSE family:

  * ``acc_at_tau`` — fraction of (step, joint) pairs inside tau, for tau in
    ``ACC_TAUS`` multiples of that joint's action std.  Per-joint rather than a flat
    radian threshold because the action vector mixes 14 arm joints in radians with
    grippers in [0, 1].  Always computed: it is one broadcast compare per batch, and
    gating something that cheap costs more to maintain than to run.
  * ``eef`` / ``eef_aggregate`` — end-effector position (m) and orientation (rad) error
    for both arms, via the same MJCF forward kinematics
    (``robot_data_platform/tool/tr_joint_to_eef.py``) that produced the EEF datasets.
    Orientation error is the geodesic angle between quaternions, not a difference of
    Euler angles, which wraps at pi and is ill-conditioned near gimbal lock.  This one
    *is* conditional, because it costs an FK pass: it turns on by itself whenever the
    MJCF resolves and the dataset's joint names cover both chains, reports why it did
    not otherwise, and is forced off with ``--no-eef``.  It is scored for the four core
    variants only, not the ``filt_*`` ablation rungs, which are a joint-space question.

Everything else (contamination fingerprinting, checkpoint-owned normalisation, the
padding mask, the streaming accumulator) is identical to the original harness.

What it still cannot reproduce
------------------------------
* **Closed loop.**  Each anchor is scored open-loop from a *demonstrated* state.  On the
  robot, chunk N+1 starts from wherever chunk N actually left the arm.  This is a
  teacher-forced segment evaluation, not a rollout.
* **The image path.**  Deployment feeds a JPEG (q90, re-encoded by ``vla_node``) split
  from the live mosaic; the dataset feeds a video-codec frame.  Worse, the deployed
  splitter (``marvain_m6_http._split_quad_image``) downscales the head tile with
  ``INTER_AREA`` while the training conversion and ``lebot_client.split_hero3_image``
  both use ``INTER_LINEAR`` — a genuine train/deploy mismatch that belongs in the deploy
  code, not here.
* **The gripper observation.**  Training state is a command echo (95 % exactly 0.0/1.0);
  deployment feeds real feedback pushed through ``gripper_state_calibration``, whose
  two endpoints were eyeballed from one rollout.  Intermediate values are off-manifold
  in a way no offline dataset contains.

Method (unchanged parts)
------------------------
For every sampled anchor frame t:
  1. Build the observation exactly as training did (same ``delta_timestamps``, same
     uint8->float/255 conversion, same preprocessor loaded *from the checkpoint*).
  2. ``policy.predict_action_chunk`` -> un-normalise with the checkpoint's postprocessor.
  3. Truncate to ``--n-action-steps`` and (for ``policy_deployed``) apply the deploy
     rewrite, anchored on the same anchor's raw ``observation.state``.
  4. Compare against the recorded action at t+L .. t+L+N-1 where L is
     ``--latency-steps``, masking steps ``action_is_pad`` marks as past the episode end.

Usage
-----
    python offline_chunk_eval.py \
        --checkpoint /mnt/robot_platform/jobs/<job>/run/checkpoints/last/pretrained_model \
        --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_1 \
        --n-action-steps 50 --stride 20 --out report.json

Self-check:  python offline_chunk_eval.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------------------
# deployment action-chunk rewrite
# --------------------------------------------------------------------------------------
#: Root of the checkout whose rollout code actually drives the robot.  The trajectory
#: helpers are loaded *by file path* rather than by importing ``lerobot.rollout``: that
#: checkout ships its own ``lerobot`` package, and importing it would shadow the
#: train-venv ``lerobot`` this script runs the policy with.
DEFAULT_VLAHOST_SRC = Path("/home/kewei/YING/lerobot_vlahost")

#: ``send_next_action_chunk``'s fixed bridge length.  Not configurable there either.
DEPLOY_BRIDGE_STEPS = 40


def load_deploy_trajectory_ops(vlahost_src: Path):
    """Import ``lerobot/rollout/trajectory.py`` from the deploy checkout, by path.

    Loading the real module rather than re-implementing it is the whole point: a
    re-implementation would drift from the code on the robot, which is exactly the class
    of bug this script exists to measure.  The module imports only ``math``,
    ``dataclasses`` and ``torch``, so it loads standalone.
    """
    path = vlahost_src / "src/lerobot/rollout/trajectory.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- point --vlahost-src at the lerobot_vlahost checkout "
            f"whose deploy.py runs on the robot"
        )
    spec = importlib.util.spec_from_file_location("_deploy_trajectory", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__] and raises AttributeError if the module is not there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The deploy filter stack, in the order ``send_next_action_chunk`` applies it.  Anything
#: here can be switched off with ``--filters``; ``gripper_clip`` is the driver's wire
#: clamp (``_prepare_action``) rather than a trajectory filter, and is listed last
#: because that is where it happens.
DEPLOY_FILTER_ORDER = (
    "rollbacks",
    "gripper_loops",
    "smoothing",
    "excursions",
    "bridge",
    "gripper_clip",
)


def apply_deploy_filter(ops, name: str, chunk: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Apply one stage of the deploy rewrite to an ``[N, A]`` chunk, returning a new tensor.

    Every call site and constant is copied verbatim from
    ``lerobot/rollout/strategies/core.py:send_next_action_chunk``; ``state`` is the raw
    (un-normalised) ``observation.state`` of the anchor, which is what ``obs_raw``
    supplies on the robot.  Nothing here mutates ``chunk`` -- the caller reuses it to
    build the other ablation variants.
    """
    if name == "rollbacks":
        return ops.remove_small_rollbacks(
            chunk, joint_count=14, window_size=10, max_rollback_steps=2
        )[0]
    if name == "gripper_loops":
        return ops.remove_open_gripper_loops(
            chunk,
            joint_count=14,
            joints_per_arm=7,
            left_gripper_index=14,
            right_gripper_index=15,
            min_excursion=math.radians(1.0),
            max_excursion=math.radians(8.0),
            max_return_gap=math.radians(0.6),
            max_return_ratio=0.2,
            max_duration_steps=30,
            open_gripper_threshold=0.1,
            gripper_margin_steps=3,
            continuation_steps=3,
        )[0]
    if name == "smoothing":
        return ops.smooth_action_chunk(chunk, joint_count=14, passes=1)
    if name == "excursions":
        return ops.smooth_large_excursions(
            chunk, joint_count=14, wave_threshold=math.radians(100.0)
        )[0]
    if name == "bridge":
        # Fixed-K real-state bridge: the first K arm steps are discarded and replaced by a
        # zero-start-velocity Hermite from the measured pose to step K-1 of the chunk.
        out = chunk.clone()
        bridge = min(DEPLOY_BRIDGE_STEPS, chunk.shape[0])
        join = bridge - 1
        for i in range(min(14, chunk.shape[1])):
            end_velocity = (
                chunk[join + 1, i] - chunk[join, i] if join + 1 < chunk.shape[0] else 0.0
            )
            out[:bridge, i] = ops.cubic_hermite_segment(
                state[i],
                chunk[join, i],
                bridge,
                start_velocity=0.0,
                end_velocity=end_velocity,
                dtype=chunk.dtype,
                device=chunk.device,
            )
        return out
    if name == "gripper_clip":
        if chunk.shape[1] < 16:
            return chunk
        out = chunk.clone()
        out[:, 14:16] = out[:, 14:16].clamp(0.0, 1.0)
        return out
    raise ValueError(f"unknown deploy filter {name!r}; known: {DEPLOY_FILTER_ORDER}")


def deploy_rewrite_chunk(
    ops, chunk: torch.Tensor, state: torch.Tensor, filters=DEPLOY_FILTER_ORDER
) -> torch.Tensor:
    """Fold the enabled filters over one chunk, always in the deploy order."""
    for name in DEPLOY_FILTER_ORDER:
        if name in filters:
            chunk = apply_deploy_filter(ops, name, chunk, state)
    return chunk


def deploy_rewrite_batch(ops, pred: torch.Tensor, state: torch.Tensor, filters) -> torch.Tensor:
    """``deploy_rewrite_chunk`` over a ``[B, N, A]`` batch -- the ops are per-chunk."""
    return torch.stack(
        [deploy_rewrite_chunk(ops, pred[b], state[b], filters) for b in range(pred.shape[0])]
    )


#: Ablation variants, keyed by the accumulator name they get in the report.  The five
#: trajectory filters are cumulative in deploy order, so the whole ladder costs one full
#: rewrite: each rung is the previous rung's output.  ``filt_4_excursions`` is therefore
#: also "everything except the bridge", and ``filt_5_bridge`` is the full deploy chunk.
ABLATION_KEYS = (
    "filt_0_clip_only",
    "filt_1_rollbacks",
    "filt_2_gripper_loops",
    "filt_3_smoothing",
    "filt_4_excursions",
    "filt_5_bridge",
    "filt_bridge_only",
)


def deploy_ablation_chunk(ops, chunk: torch.Tensor, state: torch.Tensor) -> dict:
    """One chunk -> every ablation variant, sharing the cumulative work."""
    clip = lambda c: apply_deploy_filter(ops, "gripper_clip", c, state)  # noqa: E731
    out = {"filt_0_clip_only": clip(chunk)}
    current = chunk
    for i, name in enumerate(("rollbacks", "gripper_loops", "smoothing", "excursions", "bridge"), 1):
        current = apply_deploy_filter(ops, name, current, state)
        out[ABLATION_KEYS[i]] = clip(current)
    out["filt_bridge_only"] = clip(apply_deploy_filter(ops, "bridge", chunk, state))
    return out


def deploy_ablation_batch(ops, pred: torch.Tensor, state: torch.Tensor) -> dict:
    """``deploy_ablation_chunk`` over a ``[B, N, A]`` batch."""
    per_item = [deploy_ablation_chunk(ops, pred[b], state[b]) for b in range(pred.shape[0])]
    return {k: torch.stack([d[k] for d in per_item]) for k in ABLATION_KEYS}


# --------------------------------------------------------------------------------------
# end-effector pose error
# --------------------------------------------------------------------------------------
#: MJCF whose chains ``Apex_Deploy`` drives.  ``tr_joint_to_eef.DEFAULT_MJCF`` points at a
#: sibling path that does not exist in this checkout, so the location is given here.
DEFAULT_MJCF = Path(
    "/home/kewei/YING/Apex_Deploy_new/robot_node/marvin_description/mjcf/matrix/m6_696.xml"
)
DEFAULT_EEF_FRAMES = ("left_tool", "right_tool")

#: The FK implementation that produced the EEF datasets, loaded by path.
DEFAULT_EEF_TOOL = Path("/home/kewei/YING/robot_data_platform/tool/tr_joint_to_eef.py")

#: (position metres, orientation radians) pairs.  Unlike the joint-space thresholds these
#: are absolute, because a millimetre is a millimetre regardless of the action std.
EEF_TAUS = ((0.005, 0.01), (0.01, 0.025), (0.025, 0.05), (0.05, 0.1))
EEF_CHANNELS = ["left_pos_m", "left_rot_rad", "right_pos_m", "right_rot_rad"]


class EefPoseError:
    """Forward kinematics on both arms, as a 4-channel error the accumulator can eat.

    ``tr_joint_to_eef.MjcfForwardKinematics`` is loaded by path rather than reimplemented,
    for the same reason the deploy rewrite is: it is the code that produced the EEF
    datasets, and a second copy of an FK chain is a second thing to keep in sync.

    Orientation error is the geodesic angle between quaternions, not a difference of Euler
    angles -- rpy differences wrap at pi and blow up near gimbal lock, which is exactly
    where a wrist joint spends its time.
    """

    def __init__(self, tool_path: Path, mjcf: Path, joint_names: list[str],
                 frames: tuple[str, str] = DEFAULT_EEF_FRAMES):
        spec = importlib.util.spec_from_file_location("_tr_joint_to_eef", tool_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # quaternion, not euler: see the class docstring
        self.fk = module.MjcfForwardKinematics(
            mjcf, joint_names, frames[0], frames[1], rotation_repr="quaternion"
        )

    @staticmethod
    def _resolve(tool_path: Path, mjcf: Path, joint_names: list[str], frames):
        """Return an instance, or ``None`` with a reason if this dataset cannot have one."""
        if not tool_path.is_file():
            return None, f"{tool_path} not found"
        if not mjcf.is_file():
            return None, f"{mjcf} not found"
        try:
            return EefPoseError(tool_path, mjcf, joint_names, frames), None
        except Exception as exc:  # missing joints, unknown frame, unsupported joint type
            return None, str(exc)

    def errors(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """(B, H, J) joint-space pairs -> (B, H, 4) [l_pos, l_rot, r_pos, r_rot]."""
        import numpy as np

        b, h, _ = pred.shape
        flat = torch.cat([pred, gt]).reshape(-1, pred.shape[-1]).numpy()
        left, right = self.fk.evaluate(flat)
        out = []
        for arm in (left, right):
            a, c = arm[: b * h], arm[b * h:]
            pos = np.linalg.norm(a[:, :3] - c[:, :3], axis=1)
            # |dot| because q and -q are the same rotation; clipped because the fk output
            # is float32 and a dot of 1.0000001 makes arccos nan.
            dot = np.abs((a[:, 3:] * c[:, 3:]).sum(axis=1)).clip(0.0, 1.0)
            out += [pos, 2.0 * np.arccos(dot)]
        return torch.from_numpy(np.stack(out, axis=1)).float().reshape(b, h, 4)


class EefSliceError:
    """Pose error for a policy whose action *is* an EEF pose, so no kinematics are needed.

    ``tr_joint_to_eef`` datasets carry ``[eef_l(6), eef_r(6), gripper_L, gripper_R]``; the
    pose is already there and FK would be answering a question that has been answered.
    Detected from the action names rather than the width, because 14 alone does not say
    which 14.

    Position error is the Euclidean distance between the two points, not the per-axis mean
    that ``runs/20260903_pp_eef_state_head/summarise.py:group`` reports -- ``mae_per_joint``
    already gives the per-axis view, and averaging x, y and z answers "how wrong is a
    coordinate" rather than "how far away is the gripper".  The two are not comparable:
    for isotropic Gaussian error E||e|| / E|e_i| is exactly 2 (chi_3 mean 2*sigma*sqrt(2/pi)
    over half-normal mean sigma*sqrt(2/pi)), and the measured ratio on this checkpoint is
    2.00.  A number from here is therefore ~2x the mm column in the 09-03 report.
    """

    #: ``R = Rz(yaw) Ry(pitch) Rx(roll)``, matching tr_joint_to_eef's stated convention.
    SUFFIXES = ("_x", "_y", "_z", "_roll", "_pitch", "_yaw")

    def __init__(self, groups: list[list[int]]):
        self.groups = groups  # [[6 column indices for left], [... right]]

    @classmethod
    def detect(cls, names: list[str]):
        """Return an instance if exactly two complete pose groups are present."""
        prefixes: dict[str, dict[str, int]] = {}
        for i, n in enumerate(names):
            for suf in cls.SUFFIXES:
                if n.endswith(suf):
                    prefixes.setdefault(n[: -len(suf)], {})[suf] = i
        full = [p for p, d in prefixes.items() if len(d) == len(cls.SUFFIXES)]
        if len(full) != 2:
            return None, f"expected 2 complete eef pose groups in action names, found {len(full)}"
        # sorted so left/right ordering follows the action vector, not dict insertion
        full.sort(key=lambda p: min(prefixes[p].values()))
        return cls([[prefixes[p][s] for s in cls.SUFFIXES] for p in full]), None

    def errors(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        import numpy as np

        b, h, _ = pred.shape
        a = pred.reshape(-1, pred.shape[-1]).numpy().astype(np.float64)
        c = gt.reshape(-1, gt.shape[-1]).numpy().astype(np.float64)
        out = []
        for cols in self.groups:
            pos = np.linalg.norm(a[:, cols[:3]] - c[:, cols[:3]], axis=1)
            ra, rc = _rpy_to_matrix(a[:, cols[3:]]), _rpy_to_matrix(c[:, cols[3:]])
            # geodesic angle: arccos((tr(Ra^T Rc) - 1) / 2), clipped for float error
            tr = np.einsum("nij,nij->n", ra, rc)
            out += [pos, np.arccos(((tr - 1.0) / 2.0).clip(-1.0, 1.0))]
        return torch.from_numpy(np.stack(out, axis=1)).float().reshape(b, h, 4)


def _rpy_to_matrix(rpy):
    """(N, 3) roll/pitch/yaw -> (N, 3, 3) with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    import numpy as np

    cr, cp, cy = np.cos(rpy).T
    sr, sp, sy = np.sin(rpy).T
    return np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], axis=-1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], axis=-1),
            np.stack([-sp, cp * sr, cp * cr], axis=-1),
        ],
        axis=1,
    )


def resolve_eef(names: list[str], tool: Path, mjcf: Path, frames):
    """Pick the pose-error path this action space supports, or say why there is none.

    EEF-space first: if the action already carries poses, FK is not merely unnecessary,
    it is inapplicable -- the MJCF chain expects joint angles.
    """
    obj, why_slice = EefSliceError.detect(names)
    if obj is not None:
        return obj, "eef-space (direct, no kinematics)", None
    obj, why_fk = EefPoseError._resolve(tool, mjcf, names, frames)
    if obj is not None:
        return obj, f"joint-space FK ({mjcf.name})", None
    return None, None, f"not eef-space ({why_slice}); not joint-space ({why_fk})"


def eef_accumulator(horizon: int) -> ChunkErrorAccumulator:
    """A 4-channel accumulator whose acc@tau thresholds are metres and radians."""
    thresholds = torch.tensor([[p, r, p, r] for p, r in EEF_TAUS], dtype=torch.float64)
    labels = [f"{p * 1000:g}mm+{r:g}rad" for p, r in EEF_TAUS]
    return ChunkErrorAccumulator(horizon, 4, thresholds, labels)


# --------------------------------------------------------------------------------------
# metric accumulation
# --------------------------------------------------------------------------------------
#: acc@tau thresholds, in units of each joint's action std.  Per-joint rather than a flat
#: radian value because the action vector mixes units: 14 arm joints in radians and 2
#: grippers in [0, 1], where "within 0.05" means two different things.
#:
#: These are always computed rather than hidden behind a flag.  One broadcast compare per
#: batch is cheaper than the branch that would skip it, and a metric that is always in the
#: report is one nobody has to re-run the harness to get.
ACC_TAUS = (0.1, 0.25, 0.5, 1.0)
TAU_LABELS = [f"{t:g}sigma" for t in ACC_TAUS]


class ChunkErrorAccumulator:
    """Streaming sum of |pred - gt| and (pred - gt)^2 over (horizon, joint), masked.

    Kept as running sums rather than a list of batches so memory does not grow with the
    number of anchors -- a full pass over six datasets is ~10^4 anchors x 100 x 16.
    """

    def __init__(self, horizon: int, n_joints: int, thresholds: torch.Tensor | None = None,
                 tau_labels: list[str] | None = None, device: str = "cpu"):
        self.abs_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.sq_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.count = torch.zeros(horizon, 1, dtype=torch.float64, device=device)
        # thresholds: (n_tau, n_joints) in raw action units, or None to skip acc@tau.
        # Stored as (n_tau, 1, 1, n_joints) to broadcast against a (B, H, J) error.
        self.thresholds = (
            None if thresholds is None
            else thresholds.double().to(device).view(thresholds.shape[0], 1, 1, -1)
        )
        self.hit_sum = (
            None if thresholds is None
            else torch.zeros(thresholds.shape[0], horizon, n_joints,
                             dtype=torch.float64, device=device)
        )
        # Carried on the accumulator rather than read from a module global: the joint-space
        # thresholds are multiples of sigma, the EEF ones are metres and radians, and the
        # two must not borrow each other's labels.
        self.tau_labels = tau_labels

    def update(self, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> None:
        """pred/gt: (B, H, J) float. valid: (B, H) bool -- False where the chunk ran past
        the end of the episode and the recorded action is padding."""
        err = pred.double() - gt.double()
        mask = valid.unsqueeze(-1).double()
        if self.hit_sum is not None:
            # Masked *after* the compare, not before: a padded step has err == 0, which
            # would otherwise score as a hit at every threshold.
            hit = (err.abs().unsqueeze(0) <= self.thresholds).double() * mask.unsqueeze(0)
            self.hit_sum += hit.sum(dim=1)
        err = err * mask
        self.abs_sum += err.abs().sum(dim=0)
        self.sq_sum += err.pow(2).sum(dim=0)
        self.count += valid.double().sum(dim=0).unsqueeze(-1)

    # -- reductions ---------------------------------------------------------------------
    def _safe(self, num: torch.Tensor) -> torch.Tensor:
        return num / self.count.clamp_min(1.0)

    def mae_per_horizon_joint(self) -> torch.Tensor:
        return self._safe(self.abs_sum)

    def mae_per_horizon(self) -> torch.Tensor:
        return self.mae_per_horizon_joint().mean(dim=1)

    def mae_per_joint(self, upto: int | None = None) -> torch.Tensor:
        h = slice(0, upto) if upto else slice(None)
        return self.abs_sum[h].sum(0) / self.count[h].sum().clamp_min(1.0)

    def mae(self, upto: int | None = None) -> float:
        h = slice(0, upto) if upto else slice(None)
        n = self.count[h].sum() * self.abs_sum.shape[1]
        return (self.abs_sum[h].sum() / n.clamp_min(1.0)).item()

    def rmse(self, upto: int | None = None) -> float:
        h = slice(0, upto) if upto else slice(None)
        n = self.count[h].sum() * self.sq_sum.shape[1]
        return (self.sq_sum[h].sum() / n.clamp_min(1.0)).sqrt().item()

    def acc_at_tau(self, upto: int | None = None) -> list[float]:
        """Fraction of valid (step, joint) pairs within each threshold."""
        if self.hit_sum is None:
            return []
        h = slice(0, upto) if upto else slice(None)
        n = self.count[h].sum() * self.hit_sum.shape[2]
        return (self.hit_sum[:, h].sum(dim=(1, 2)) / n.clamp_min(1.0)).tolist()

    def acc_at_tau_per_joint(self) -> torch.Tensor:
        """(n_tau, n_joints) -- which joints the chunk actually lands on."""
        return self.hit_sum.sum(dim=1) / self.count.sum().clamp_min(1.0)

    def n_anchor_steps(self) -> int:
        return int(self.count.sum().item())


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
def load_policy(checkpoint: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = str(checkpoint)
    cfg.device = device

    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(device)
    policy.eval()

    # The processors are loaded *from the checkpoint*, so the normalisation statistics are
    # the ones the model was trained with. Recomputing them on the eval set would leak
    # eval-set statistics into the model's inputs and flatter the result.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, cfg


def episode_action_hashes(root: Path) -> dict[int, str]:
    """SHA1 of each episode's raw action array, as an identity fingerprint.

    Needed because the dataset names in this corpus lie about provenance: the batches are
    *cumulative* merges, so `batch_5` and `batch_6` are wholly contained in the training
    set `batch_success_361` and `batch_4` is 38% contained, despite being separate
    directories with different episode counts. Matching on episode counts or on episode
    lengths finds false negatives; only the data itself is conclusive.
    """
    import glob

    import numpy as np
    import pandas as pd

    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    out = {}
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(np.float32)
        out[int(ep)] = hashlib.sha1(a.tobytes()).hexdigest()
    return out


def build_dataset(root: Path, repo_id: str, cfg, exclude_hashes: set[str] | None, invert: bool = False):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    # Same helper the training script uses, so the batch layout is identical to training.
    delta_timestamps = resolve_delta_timestamps(cfg, meta)

    episodes = None
    n_dropped = 0
    if exclude_hashes:
        fp = episode_action_hashes(root)
        # invert=True keeps ONLY the contaminated episodes. That is the within-session
        # control: scoring the seen and unseen halves of one recording batch holds the
        # session, lighting and object layout fixed, so the difference between them is
        # memorisation alone rather than memorisation plus distribution shift.
        episodes = sorted(
            ep for ep, h in fp.items() if (h in exclude_hashes) == invert
        )
        n_dropped = len(fp) - len(episodes)
        if not episodes:
            return None, meta, n_dropped

    ds = LeRobotDataset(
        repo_id=repo_id, root=root, episodes=episodes, delta_timestamps=delta_timestamps
    )
    return ds, meta, n_dropped


def action_stats(postprocessor) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-set action mean/std, read out of the checkpoint's unnormaliser."""
    from lerobot.utils.constants import ACTION

    for step in postprocessor.steps:
        stats = getattr(step, "stats", None)
        if stats and ACTION in stats:
            s = stats[ACTION]
            as_t = lambda x: torch.as_tensor(x, dtype=torch.float32).flatten().cpu()  # noqa: E731
            return as_t(s["mean"]), as_t(s["std"])
    raise RuntimeError("no action mean/std found in the postprocessor pipeline")


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    dataset_roots: list[Path],
    stride: int,
    batch_size: int,
    num_workers: int,
    device: str,
    max_anchors_per_dataset: int | None,
    train_root: list[Path],
    n_action_steps: int,
    latency_steps: int,
    vlahost_src: Path,
    filters: tuple[str, ...],
    ablation: bool,
    invert_filter: bool = False,
    dump_traces: Path | None = None,
    trace_anchors: int = 40,
    trace_episodes: set[int] | None = None,
    seed: int = 0,
    seed_repeat: int = 0,
    eef_tool: Path | None = None,
    eef_mjcf: Path = DEFAULT_MJCF,
    eef: bool = True,
) -> dict:
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    policy, preprocessor, postprocessor, cfg = load_policy(checkpoint, device)
    # `chunk_size` is ACT's name for it; patch_policy calls the same thing
    # `action_chunk_size` and has no `chunk_size` attribute at all.
    chunk_size = getattr(cfg, "chunk_size", None) or cfg.action_chunk_size
    # Index of delta 0 inside batch[ACTION]. 0 for ACT (`action_delta_indices` starts at 0),
    # n_obs_steps - 1 for patch_policy, whose action window reaches back over the history.
    a_off = -min(list(cfg.action_delta_indices or [0]) + [0])
    is_patch = cfg.type == "patch_policy"
    # The executed horizon, not the emitted one: deploy dispatches only the first
    # inference.n_action_steps rows of each chunk and replans after them.
    horizon = min(n_action_steps, chunk_size)
    if latency_steps + horizon > chunk_size:
        raise ValueError(
            f"--latency-steps {latency_steps} + --n-action-steps {horizon} exceeds the "
            f"policy chunk_size {chunk_size}; there is no ground truth that far out"
        )
    ops = load_deploy_trajectory_ops(vlahost_src)
    print(
        f"executed horizon: {horizon}/{chunk_size} steps, latency offset {latency_steps}, "
        f"deploy rewrite from {vlahost_src}",
        flush=True,
    )
    print(
        f"policy {cfg.type}: n_obs_steps={getattr(cfg, 'n_obs_steps', 1)}, "
        f"action offset {a_off}, seed {seed}"
        # only the patch path re-samples; --seed-repeat is a no-op for a deterministic head
        + (f", {seed_repeat} extra sampling seeds" if is_patch and seed_repeat else ""),
        flush=True,
    )
    print(f"policy_deployed filters: {', '.join(filters) if filters else '(none)'}"
          f"{'  + per-filter ablation' if ablation else ''}", flush=True)

    exclude_hashes = None
    if train_root:
        exclude_hashes = set()
        for tr in train_root:
            h = set(episode_action_hashes(tr).values())
            exclude_hashes |= h
            print(f"contamination filter: {len(h)} training episodes fingerprinted "
                  f"from {tr.name}", flush=True)
        print(f"contamination filter: {len(exclude_hashes)} unique training episodes total",
              flush=True)
    a_mean, a_std = action_stats(postprocessor)
    # (n_tau, J): tau_k * std_j.  Built here because a_std is the only per-joint scale the
    # checkpoint carries; a raw-unit threshold would not survive the rad/gripper mix.
    tau_thresholds = torch.as_tensor(ACC_TAUS).double().view(-1, 1) * a_std.double().view(1, -1)

    per_dataset: dict[str, dict] = {}
    keys = ["policy_raw", "policy_deployed", "hold_state", "train_mean"]
    if ablation:
        keys += list(ABLATION_KEYS)
    if is_patch and seed_repeat:
        keys += [f"seed_{r + 1}" for r in range(seed_repeat)]
    overall = {k: None for k in keys}
    # EEF error is scored only for these: the filt_* rungs exist to attribute a joint-space
    # delta to one filter, and running FK on all of them multiplies the cost of the pass
    # for a number that answers a different question.
    # The seed draws are in: without a pose-error spread between two draws of the same
    # checkpoint there is no floor to read the pose-error *differences* against, which is
    # the same argument that put seed_repeat in the joint-space table.
    eef_keys = [k for k in keys
                if k in ("policy_raw", "policy_deployed", "hold_state", "train_mean")
                or k.startswith("seed_")]
    overall_eef: dict[str, ChunkErrorAccumulator] = {}
    eef_fk = None
    eef_skip_reason = None if eef else "disabled with --no-eef"
    n_joints = None
    joint_names = None
    t_start = time.time()
    # Raw per-anchor predictions, kept only for the first scored dataset and only for the
    # first `trace_anchors` anchors (or only for the requested `trace_episodes`): the
    # accumulator above is streaming by design, so the arrays a trajectory plot needs do
    # not otherwise survive the loop.
    traces: dict[str, list] | None = {} if dump_traces else None
    traces_n = 0

    for root in dataset_roots:
        repo_id = f"{root.parent.name}/{root.name}"
        ds, meta, n_dropped = build_dataset(root, repo_id, cfg, exclude_hashes, invert_filter)
        if ds is None:
            print(f"[{repo_id}] SKIPPED - all {n_dropped} episodes are in the training set", flush=True)
            per_dataset[repo_id] = {"skipped": "fully contained in training set",
                                    "episodes_dropped": n_dropped}
            continue
        if n_joints is None:
            n_joints = meta.features[ACTION]["shape"][0]
            joint_names = meta.features[ACTION].get("names") or [f"j{i}" for i in range(n_joints)]
            for k in overall:
                overall[k] = ChunkErrorAccumulator(horizon, n_joints, tau_thresholds, TAU_LABELS)
            # Auto-enabled: if the MJCF resolves and this dataset's joint names cover both
            # chains, EEF error is reported; otherwise the run says why and carries on.
            if eef:
                eef_fk, eef_how, eef_skip_reason = resolve_eef(
                    joint_names, eef_tool or DEFAULT_EEF_TOOL, eef_mjcf, DEFAULT_EEF_FRAMES
                )
            if eef_fk is not None:
                overall_eef = {k: eef_accumulator(horizon) for k in eef_keys}
                print(f"eef pose error: {eef_how}", flush=True)
            else:
                print(f"eef pose error: skipped ({eef_skip_reason})", flush=True)

        idx = list(range(0, len(ds), stride))
        if max_anchors_per_dataset:
            idx = idx[:max_anchors_per_dataset]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
        )

        acc = {k: ChunkErrorAccumulator(horizon, n_joints, tau_thresholds, TAU_LABELS) for k in overall}
        acc_eef = {k: eef_accumulator(horizon) for k in overall_eef}
        camera_keys = meta.camera_keys
        t0 = time.time()

        for batch in loader:
            # Ground truth and state are snapshotted BEFORE the preprocessor runs: the
            # normaliser rewrites batch[ACTION] in place, and we want raw joint units.
            # a_off drops the actions that belong to the *history* frames: for
            # patch_policy the window starts at t-(n_obs_steps-1), and only the tail from
            # delta 0 on is the chunk the newest observation is responsible for.
            gt_full = batch[ACTION][:, a_off:].clone().float()  # (B, chunk_size, J)
            state = batch[OBS_STATE].clone().float()           # (B, J) or (B, S, J)
            if state.ndim == 3:
                state = state[:, -1]                           # anchor = newest frame
            valid_full = ~batch["action_is_pad"][:, a_off:]    # (B, chunk_size)
            # The robot starts executing latency_steps ticks after the observation was
            # taken (HTTP round trip + inference + vlahost's prepended blend waypoint),
            # so step k of the chunk lands on demonstrated action t+latency+k.
            lo, hi = latency_steps, latency_steps + horizon
            gt = gt_full[:, lo:hi]
            valid = valid_full[:, lo:hi]

            # Mirror lerobot_train.py's eval path: uint8 frames are scaled to [0,1]
            # before the normaliser sees them.
            for cam in camera_keys:
                if cam in batch and batch[cam].dtype == torch.uint8:
                    batch[cam] = batch[cam].to(dtype=torch.float32) / 255.0

            processed = preprocessor(batch)
            if is_patch:
                # `PatchPolicy.predict_action_chunk` rebuilds its input from the
                # `select_action` queues, which a batched offline pass never fills. The
                # model-level entry point takes the batch, and wants the cameras stacked
                # on a new axis exactly as `select_action` stacks them.
                processed[OBS_IMAGES] = torch.stack(
                    [processed[k] for k in cfg.image_features], dim=-4
                )
                torch.manual_seed(seed)
                pred = policy.model.predict(processed)
            else:
                pred = policy.predict_action_chunk(processed)  # normalised, (B, C, J)
            pred = postprocessor(pred).float().cpu()           # raw joint units
            pred = pred[:, :horizon]                           # only this much is sent
            seed_chunks: dict[str, torch.Tensor] = {}
            if is_patch and seed_repeat:
                # Same observation, different diffusion noise. The spread between draws is
                # the floor below which any difference between checkpoints is sampling.
                for r in range(seed_repeat):
                    torch.manual_seed(seed + 1 + r)
                    alt = postprocessor(policy.model.predict(processed)).float().cpu()
                    acc[f"seed_{r + 1}"].update(alt[:, :horizon], gt, valid)
                    seed_chunks[f"seed_{r + 1}"] = alt[:, :horizon]
            # what /action_chunk gets, under the requested filter set
            deployed = deploy_rewrite_batch(ops, pred, state, filters)

            acc["policy_raw"].update(pred, gt, valid)
            acc["policy_deployed"].update(deployed, gt, valid)
            if ablation:
                for k, variant in deploy_ablation_batch(ops, pred, state).items():
                    acc[k].update(variant, gt, valid)
            acc["hold_state"].update(state.unsqueeze(1).expand_as(gt), gt, valid)
            acc["train_mean"].update(
                a_mean.view(1, 1, -1).expand_as(gt), gt, valid
            )

            if acc_eef:
                # gt is the zero of an error magnitude, so the accumulator's |pred - gt|
                # and (pred - gt)^2 become the mean and RMS of the pose error itself.
                zero = None
                for k, chunk in [
                    ("policy_raw", pred),
                    ("policy_deployed", deployed),
                    ("hold_state", state.unsqueeze(1).expand_as(gt)),
                    ("train_mean", a_mean.view(1, 1, -1).expand_as(gt)),
                ] + list(seed_chunks.items()):
                    if k not in acc_eef:
                        continue
                    err = eef_fk.errors(chunk, gt)
                    if zero is None:
                        zero = torch.zeros_like(err)
                    acc_eef[k].update(err, zero, valid)

            if traces is not None and traces_n < trace_anchors:
                if trace_episodes is not None:
                    keep = torch.isin(
                        batch["episode_index"].cpu(), torch.tensor(sorted(trace_episodes))
                    )
                    n_keep = int(keep.sum())
                else:
                    keep, n_keep = None, len(gt)
                if n_keep:
                    for key, val in (
                        ("pred", deployed), ("pred_raw", pred),
                        ("gt", gt), ("state", state), ("valid", valid),
                        ("episode_index", batch["episode_index"]), ("frame_index", batch["frame_index"]),
                    ):
                        arr = val.cpu().numpy()
                        traces.setdefault(key, []).append(
                            arr[keep.numpy()] if keep is not None else arr
                        )
                    traces_n += n_keep

        if traces:
            import numpy as np
            dump_traces.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dump_traces,
                joint_names=np.array(joint_names),
                repo_id=repo_id,
                **{k: np.concatenate(v)[:trace_anchors] for k, v in traces.items()},
            )
            print(f"wrote {dump_traces}  ({traces_n} anchors from {repo_id})", flush=True)
            traces = None

        for k in overall:
            overall[k].abs_sum += acc[k].abs_sum
            overall[k].sq_sum += acc[k].sq_sum
            overall[k].count += acc[k].count
            overall[k].hit_sum += acc[k].hit_sum
        for k in overall_eef:
            overall_eef[k].abs_sum += acc_eef[k].abs_sum
            overall_eef[k].sq_sum += acc_eef[k].sq_sum
            overall_eef[k].count += acc_eef[k].count
            overall_eef[k].hit_sum += acc_eef[k].hit_sum

        per_dataset[repo_id] = {
            "episodes_evaluated": meta.total_episodes - n_dropped,
            "episodes_dropped_as_contaminated": n_dropped,
            "frames": meta.total_frames,
            "anchors": len(idx),
            "anchor_action_steps": acc["policy_raw"].n_anchor_steps(),
            "seconds": round(time.time() - t0, 1),
            **{k: summarise(acc[k], a_std, horizon, joint_names) for k in overall},
            **({"eef": {k: summarise_eef(acc_eef[k], horizon) for k in acc_eef}}
               if acc_eef else {}),
        }
        print(
            f"[{repo_id}] {len(idx)} anchors  "
            f"raw={acc['policy_raw'].mae():.5f}  "
            f"deployed={acc['policy_deployed'].mae():.5f}  "
            f"hold_state={acc['hold_state'].mae():.5f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    if n_joints is None:
        raise RuntimeError(
            "nothing was scored: every --dataset-root is fully contained in --train-root "
            "(pass --keep-only-contaminated to score the seen episodes on purpose)"
        )
    return {
        "checkpoint": str(checkpoint),
        "policy_type": cfg.type,
        "chunk_size": chunk_size,
        "executed_horizon": horizon,
        "latency_steps": latency_steps,
        "deploy_filters": list(filters),
        "deploy_filter_ablation": ablation,
        "deploy_rewrite": "core.send_next_action_chunk (rollbacks, gripper loops, "
                          f"smoothing, excursions, K={DEPLOY_BRIDGE_STEPS} Hermite bridge, "
                          "gripper clip)",
        "vlahost_src": str(vlahost_src),
        "policy_n_action_steps": getattr(cfg, "n_action_steps", None),
        "policy_n_obs_steps": getattr(cfg, "n_obs_steps", 1),
        "policy_action_head": getattr(cfg, "action_head", None),
        "policy_use_robot_state": getattr(cfg, "use_robot_state", None),
        "policy_vision_encoder": getattr(cfg, "vision_encoder", None),
        "action_offset": a_off,
        "seed": seed,
        "seed_repeat": seed_repeat,
        "joint_names": joint_names,
        "stride": stride,
        # ponytail: the diffusion seed is re-set per batch, so an anchor's sampling noise
        # depends on its index within its batch -- two runs are only comparable at the same
        # batch_size. Recorded rather than fixed; per-anchor seeding costs per-anchor predict.
        "batch_size": batch_size,
        "device": device,
        "total_seconds": round(time.time() - t_start, 1),
        "per_dataset": per_dataset,
        "aggregate": {k: summarise(overall[k], a_std, horizon, joint_names) for k in overall},
        "eef_aggregate": ({k: summarise_eef(overall_eef[k], horizon) for k in overall_eef}
                          if overall_eef else {"skipped": eef_skip_reason}),
    }


def summarise_eef(acc: ChunkErrorAccumulator, horizon: int) -> dict:
    """Per-channel only.  There is deliberately no scalar aggregate: averaging a metre
    against a radian produces a number that moves for reasons nobody can name."""
    per_h = acc.mae_per_horizon_joint()
    cuts = [c for c in (1, 10, 25, 50, horizon) if c <= horizon]
    rms = (acc._safe(acc.sq_sum).mean(dim=0)).sqrt()
    return {
        "mean_per_channel": dict(zip(EEF_CHANNELS, [round(v, 6) for v in per_h.mean(0).tolist()])),
        "rms_per_channel": dict(zip(EEF_CHANNELS, [round(v, 6) for v in rms.tolist()])),
        "mean_at_horizon": {
            str(c): dict(zip(EEF_CHANNELS, [round(v, 6) for v in per_h[:c].mean(0).tolist()]))
            for c in cuts
        },
        "mean_per_horizon": {
            ch: [round(v, 6) for v in per_h[:, i].tolist()]
            for i, ch in enumerate(EEF_CHANNELS)
        },
        "acc_at_tau": {
            t: dict(zip(EEF_CHANNELS, [round(v, 6) for v in row]))
            for t, row in zip(acc.tau_labels, acc.acc_at_tau_per_joint().tolist())
        },
        "anchor_action_steps": acc.n_anchor_steps(),
    }


def summarise(acc: ChunkErrorAccumulator, a_std: torch.Tensor, horizon: int, names: list[str]) -> dict:
    """Raw-unit and std-normalised reductions of one accumulator."""
    mae_hj = acc.mae_per_horizon_joint()
    norm_hj = mae_hj / a_std.double().clamp_min(1e-8)
    # Per-joint normalisation before the sqrt, so norm_rmse / norm_mae is a scale-free
    # tail ratio comparable to 1.2533 (normal) / 1.4142 (Laplace).
    norm_rmse = (acc._safe(acc.sq_sum) / a_std.double().clamp_min(1e-8).pow(2)).mean().sqrt().item()
    norm_mae = norm_hj.mean().item()
    cuts = [c for c in (1, 10, 25, 50, horizon) if c <= horizon]
    tau_names = acc.tau_labels or []
    out = {
        "mae": acc.mae(),
        "rmse": acc.rmse(),
        "norm_mae": norm_mae,
        "norm_rmse": norm_rmse,
        "tail_ratio": norm_rmse / max(norm_mae, 1e-12),
        "mae_at_horizon": {str(c): acc.mae(upto=c) for c in cuts},
        "norm_mae_at_horizon": {str(c): norm_hj[:c].mean().item() for c in cuts},
        "mae_per_horizon": [round(v, 6) for v in acc.mae_per_horizon().tolist()],
        "mae_per_joint": dict(zip(names, [round(v, 6) for v in acc.mae_per_joint().tolist()])),
        "norm_mae_per_joint": dict(zip(names, [round(v, 6) for v in norm_hj.mean(0).tolist()])),
        "anchor_action_steps": acc.n_anchor_steps(),
    }
    if acc.hit_sum is not None:
        out |= {
            "acc_at_tau": dict(zip(tau_names, [round(v, 6) for v in acc.acc_at_tau()])),
            "acc_at_tau_at_horizon": {
                str(c): dict(zip(tau_names, [round(v, 6) for v in acc.acc_at_tau(upto=c)]))
                for c in cuts
            },
            "acc_at_tau_per_joint": {
                t: dict(zip(names, [round(v, 6) for v in row]))
                for t, row in zip(tau_names, acc.acc_at_tau_per_joint().tolist())
            },
        }
    return out


# --------------------------------------------------------------------------------------
def selftest() -> None:
    """The accumulator is the only non-trivial logic here; check it against a hand case."""
    acc = ChunkErrorAccumulator(horizon=3, n_joints=2)
    pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]]])   # (1,3,2)
    gt = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])                    # last step is padding
    acc.update(pred, gt, valid)

    # padded step must contribute nothing, to neither the sum nor the count
    assert acc.count.flatten().tolist() == [1.0, 1.0, 0.0], acc.count
    assert abs(acc.mae_per_horizon()[2].item()) < 1e-12, "padding leaked into the mean"
    # mean over the two valid steps and two joints: (1+1+2+2)/4 = 1.5
    assert abs(acc.mae() - 1.5) < 1e-12, acc.mae()
    # horizon-1 cut sees only the first step: (1+1)/2 = 1.0
    assert abs(acc.mae(upto=1) - 1.0) < 1e-12, acc.mae(upto=1)
    # rmse over valid: sqrt((1+1+4+4)/4) = sqrt(2.5)
    assert abs(acc.rmse() - 2.5**0.5) < 1e-12, acc.rmse()

    # two batches must accumulate exactly like one concatenated batch
    a, b = ChunkErrorAccumulator(2, 1), ChunkErrorAccumulator(2, 1)
    p1, g1 = torch.tensor([[[1.0], [3.0]]]), torch.zeros(1, 2, 1)
    p2, g2 = torch.tensor([[[5.0], [7.0]]]), torch.zeros(1, 2, 1)
    v = torch.ones(1, 2, dtype=torch.bool)
    a.update(p1, g1, v)
    a.update(p2, g2, v)
    b.update(torch.cat([p1, p2]), torch.cat([g1, g2]), torch.cat([v, v]))
    assert torch.allclose(a.mae_per_horizon(), b.mae_per_horizon())
    assert abs(a.mae() - 4.0) < 1e-12, a.mae()

    # with unit a_std and no padding, the normalised pair must collapse onto the raw pair
    s = summarise(b, torch.ones(1), horizon=2, names=["j0"])
    assert abs(s["norm_mae"] - s["mae"]) < 1e-12, s
    assert abs(s["norm_rmse"] - s["rmse"]) < 1e-12, s
    assert abs(s["tail_ratio"] - 21**0.5 / 4.0) < 1e-12, s["tail_ratio"]
    assert "acc_at_tau" not in s, "tau keys must be absent when no thresholds were given"

    # acc@tau: the padded step must not score as a hit despite its zero error.
    tau = torch.tensor([[0.5, 0.5], [1.5, 1.5]])            # (n_tau=2, n_joints=2)
    t = ChunkErrorAccumulator(horizon=3, n_joints=2, thresholds=tau, tau_labels=["a", "b"])
    t.update(pred, gt, valid)                                # errors 1,1 | 2,2 | 9,9(pad)
    # tau=0.5 catches nothing; tau=1.5 catches only the two step-0 joints, out of the
    # 4 valid (step, joint) pairs.  A padded hit would make these 2/6 and 4/6.
    assert t.acc_at_tau() == [0.0, 0.5], t.acc_at_tau()
    assert t.acc_at_tau(upto=1) == [0.0, 1.0], t.acc_at_tau(upto=1)
    assert t.acc_at_tau_per_joint().tolist() == [[0.0, 0.0], [0.5, 0.5]]
    # and it must stay batch-split invariant, like the other two sums
    u, w = (ChunkErrorAccumulator(3, 2, thresholds=tau, tau_labels=["a", "b"]) for _ in range(2))
    u.update(pred, gt, valid)
    u.update(pred * 0.1, gt, valid)
    w.update(torch.cat([pred, pred * 0.1]), torch.cat([gt, gt]), torch.cat([valid, valid]))
    assert u.acc_at_tau() == w.acc_at_tau(), (u.acc_at_tau(), w.acc_at_tau())
    print("selftest OK (accumulator)")


def selftest_eef_slice() -> None:
    """The EEF-space path: no kinematics, but the angle metric still has to be an angle."""
    names = [f"eef_{s}_{c}" for s in "lr" for c in ("x", "y", "z", "roll", "pitch", "yaw")]
    names += ["gripper_L", "gripper_R"]
    obj, how, why = resolve_eef(names, Path("/nonexistent.py"), Path("/nonexistent.xml"),
                                DEFAULT_EEF_FRAMES)
    assert obj is not None and "eef-space" in how, why
    assert obj.groups == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], obj.groups

    gt = torch.zeros(2, 3, 14)
    assert torch.allclose(obj.errors(gt.clone(), gt), torch.zeros(2, 3, 4), atol=1e-6)

    # a 30 mm offset on left x and 0.1 rad of left yaw, on one (anchor, step) only
    pred = gt.clone()
    pred[0, 1, 0], pred[0, 1, 5] = 0.03, 0.1
    err = obj.errors(pred, gt)
    assert abs(err[0, 1, 0] - 0.03) < 1e-6 and abs(err[0, 1, 1] - 0.1) < 1e-6, err[0, 1]
    assert err[0, 1, 2:].abs().max() < 1e-6, "left leaked into right"
    assert err[0, 0].abs().max() < 1e-6 and err[1].abs().max() < 1e-6, "leaked across reshape"

    # position is the Euclidean distance, not the per-axis mean: 3-4-5 on x/y
    d = gt.clone()
    d[0, 0, 0], d[0, 0, 1] = 0.03, 0.04
    assert abs(obj.errors(d, gt)[0, 0, 0] - 0.05) < 1e-6, "position must be a distance"

    # the whole reason rpy is not subtracted: yaw pi and -pi are the same rotation
    a, b = gt.clone(), gt.clone()
    a[..., 5], b[..., 5] = math.pi, -math.pi
    assert obj.errors(a, b)[0, 0, 1] < 1e-5, "euler wraparound leaked into the angle"
    # and a rotation must never exceed pi
    r = torch.rand(4, 3, 14) * 8 - 4
    assert obj.errors(r, torch.rand(4, 3, 14) * 8 - 4)[..., [1, 3]].max() <= math.pi + 1e-6
    print("selftest OK (eef slice)")


def selftest_eef(tool: Path = DEFAULT_EEF_TOOL, mjcf: Path = DEFAULT_MJCF) -> None:
    """Check the FK wrapper on invariants that hold whatever the chain geometry is."""
    names = [f"Joint{i}_{s}" for s in "LR" for i in range(1, 8)] + ["gripper_L", "gripper_R"]
    fk, why = EefPoseError._resolve(tool, mjcf, names, DEFAULT_EEF_FRAMES)
    if fk is None:
        print(f"selftest SKIPPED (eef): {why}")
        return

    b, h, j = 2, 3, 16
    gt = torch.zeros(b, h, j)
    # identical chunks -> exactly zero on all four channels
    assert torch.allclose(fk.errors(gt.clone(), gt), torch.zeros(b, h, 4), atol=1e-6)

    # A left-arm joint must move the left channels and leave the right ones at zero.
    # Joint2_L rather than Joint1_L: at the zero configuration the tool frame lies *on*
    # the Joint1 axis, so rolling it turns the gripper in place for no position error.
    pred = gt.clone()
    pred[0, 1, 1] = 0.3                      # Joint2_L on one (anchor, step) only
    err = fk.errors(pred, gt)
    assert err[0, 1, 0] > 1e-3, err[0, 1]
    assert abs(err[0, 1, 1] - 0.3) < 1e-4, "a single hinge must rotate the tool by its angle" 
    assert torch.allclose(err[0, 1, 2:], torch.zeros(2), atol=1e-6), "left leaked into right"
    # and it must not leak across the batch/horizon reshape
    assert torch.allclose(err[0, 0], torch.zeros(4), atol=1e-6), "leaked across horizon"
    assert torch.allclose(err[1], torch.zeros(h, 4), atol=1e-6), "leaked across batch"

    # gripper columns are not on either chain, so they cannot move a pose
    g = gt.clone()
    g[..., 14:] = 1.0
    assert torch.allclose(fk.errors(g, gt), torch.zeros(b, h, 4), atol=1e-6)

    # a pi rotation about a wrist joint must give a finite angle, not nan from arccos
    w = gt.clone()
    w[..., 6] = math.pi
    assert torch.isfinite(fk.errors(w, gt)).all(), "arccos domain not clipped"

    # the 4-channel accumulator carries metre/radian labels, not sigma ones
    a = eef_accumulator(h)
    a.update(err, torch.zeros_like(err), torch.ones(b, h, dtype=torch.bool))
    out = summarise_eef(a, h)
    assert set(out["mean_per_channel"]) == set(EEF_CHANNELS), out["mean_per_channel"]
    assert "mm" in next(iter(out["acc_at_tau"])), out["acc_at_tau"]
    print("selftest OK (eef pose error)")


def parse_filters(spec: str) -> tuple[str, ...]:
    """``"all"`` / ``"none"`` / a comma list of DEPLOY_FILTER_ORDER names."""
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return DEPLOY_FILTER_ORDER
    if spec == "none":
        return ()
    names = tuple(n.strip() for n in spec.split(",") if n.strip())
    unknown = [n for n in names if n not in DEPLOY_FILTER_ORDER]
    if unknown:
        raise ValueError(
            f"unknown filter(s) {unknown}; known: {', '.join(DEPLOY_FILTER_ORDER)} "
            f"(or 'all' / 'none')"
        )
    # Order is not the caller's to choose: the deploy stack applies these in a fixed
    # sequence and smoothing-then-bridge is not the same chunk as bridge-then-smoothing.
    return tuple(n for n in DEPLOY_FILTER_ORDER if n in names)


def selftest_deploy(vlahost_src: Path = DEFAULT_VLAHOST_SRC) -> None:
    """The deploy rewrite is the other non-trivial path; check its load-bearing effects.

    Not a re-derivation of the filters (they are the robot's own code, imported) -- just
    that this wrapper wires them up the way ``send_next_action_chunk`` does.
    """
    ops = load_deploy_trajectory_ops(vlahost_src)
    n, a = 50, 16
    # A ramp on every arm joint, well away from the current pose, plus a closed gripper.
    chunk = torch.zeros(n, a)
    for j in range(14):
        chunk[:, j] = torch.linspace(1.0, 2.0, n)
    chunk[:, 14] = 1.5      # out of range: must be clipped to 1.0
    chunk[:, 15] = -0.2     # out of range: must be clipped to 0.0
    state = torch.full((a,), 0.5)

    out = deploy_rewrite_chunk(ops, chunk, state)
    assert out.shape == chunk.shape, out.shape
    # The bridge starts at the *measured* pose, not at the policy's first action.
    assert torch.allclose(out[0, :14], state[:14], atol=1e-5), out[0, :14]
    assert abs(float(chunk[0, 0]) - 1.0) < 1e-9, "input chunk must not be mutated in place"
    # ... and rejoins the (smoothed) chunk at step K-1.
    join = DEPLOY_BRIDGE_STEPS - 1
    assert abs(float(out[join, 0]) - float(chunk[join, 0])) < 5e-3, (out[join, 0], chunk[join, 0])
    # Zero start velocity: the first step is far smaller than a uniform ramp would give.
    assert float(out[1, 0] - out[0, 0]) < float(out[join, 0] - out[0, 0]) / join
    # Steps past the bridge are policy output (smoothing leaves interior ramps alone).
    assert abs(float(out[-1, 0]) - 2.0) < 1e-5, out[-1, 0]
    # Grippers: clipped, never bridged.
    assert float(out[:, 14].max()) <= 1.0 and float(out[:, 15].min()) >= 0.0

    # --- filter selection -------------------------------------------------------------
    assert parse_filters("all") == DEPLOY_FILTER_ORDER
    assert parse_filters("none") == ()
    # typed order must not change application order
    assert parse_filters("bridge,rollbacks") == ("rollbacks", "bridge")
    try:
        parse_filters("smoothing,nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown filter name was accepted")

    # No filters = the policy's own chunk, untouched.
    assert torch.equal(deploy_rewrite_chunk(ops, chunk, state, ()), chunk)
    # Turning the bridge off leaves the opening as policy output ...
    no_bridge = deploy_rewrite_chunk(
        ops, chunk, state,
        parse_filters("rollbacks,gripper_loops,smoothing,excursions,gripper_clip"),
    )
    assert abs(float(no_bridge[0, 0]) - 1.0) < 1e-5, no_bridge[0, 0]
    # ... and the ablation ladder's 4th rung is exactly that chunk, while its 5th rung is
    # the full rewrite -- the property that makes the ladder cost one pass, not seven.
    variants = deploy_ablation_chunk(ops, chunk, state)
    assert set(variants) == set(ABLATION_KEYS), sorted(variants)
    assert torch.allclose(variants["filt_4_excursions"], no_bridge, atol=1e-6)
    assert torch.allclose(variants["filt_5_bridge"], out, atol=1e-6)
    assert torch.equal(variants["filt_0_clip_only"],
                       apply_deploy_filter(ops, "gripper_clip", chunk, state))
    print("selftest OK (deploy rewrite + filter selection)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--dataset-root", type=Path, action="append", default=[])
    p.add_argument("--stride", type=int, default=20, help="sample every Nth frame as a chunk anchor")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-anchors-per-dataset", type=int, default=None)
    p.add_argument("--train-root", type=Path, action="append", default=[],
                   help="training dataset; its episodes are fingerprinted and any\neval episode with matching action data is dropped")
    p.add_argument("--n-action-steps", type=int, default=50,
                   help="executed horizon: how many steps of each chunk deploy actually\n"
                        "dispatches (inference.n_action_steps in the deploy config)")
    p.add_argument("--latency-steps", type=int, default=0,
                   help="shift ground truth by N ticks to account for the delay between\n"
                        "the observation and the first executed setpoint (HTTP round\n"
                        "trip + inference + vlahost's prepended blend waypoint).\n"
                        "Sensitivity knob; 0 reproduces the original harness's alignment")
    p.add_argument("--vlahost-src", type=Path, default=DEFAULT_VLAHOST_SRC,
                   help="checkout whose rollout/trajectory.py runs on the robot")
    p.add_argument("--filters", default="none",
                   help="which deploy filters policy_deployed goes through:\n"
                        "'none' (default), 'all', or a comma list of\n"
                        + ", ".join(DEPLOY_FILTER_ORDER)
                        + ".\nApplied in that fixed order whatever order you type.")
    p.add_argument("--filter-ablation", action="store_true",
                   help="also score the filters cumulatively (clip only, +rollbacks,\n"
                        "+gripper_loops, +smoothing, +excursions, +bridge) plus the\n"
                        "bridge alone, so each stage's cost is attributable. Shares the\n"
                        "cumulative work, so it is roughly free")
    p.add_argument("--keep-only-contaminated", action="store_true",
                   help="invert --train-root: score ONLY episodes that ARE in the\ntraining set (within-session control)")
    p.add_argument("--out", type=Path)
    p.add_argument("--dump-traces", type=Path, default=None,
                   help="save raw per-anchor pred/gt/state for the first scored dataset to an .npz, "
                        "for plot_traces.py")
    p.add_argument("--trace-anchors", type=int, default=200,
                   help="max anchors to keep in the trace dump")
    p.add_argument("--trace-episode", type=int, action="append", default=[],
                   help="only dump anchors from this episode index (repeatable). Without it the "
                        "first --trace-anchors anchors of the dataset are kept, which with "
                        "stride>1 all come from the first episode -- pass --trace-episode N "
                        "(and raise --trace-anchors if N's episode has more anchors) to plot "
                        "a specific episode")
    p.add_argument("--seed", type=int, default=0,
                   help="torch seed for the diffusion head's sampling noise (patch_policy only)")
    p.add_argument("--seed-repeat", type=int, default=0,
                   help="re-run each anchor with N further seeds and score them separately,\n"
                        "to size the sampling noise against the differences being measured")
    p.add_argument("--no-eef", dest="eef", action="store_false",
                   help="skip end-effector pose error even when the MJCF resolves; by "
                        "default it is computed whenever the joint names cover both chains")
    p.add_argument("--eef-tool", type=Path, default=DEFAULT_EEF_TOOL,
                   help="tr_joint_to_eef.py providing MjcfForwardKinematics")
    p.add_argument("--eef-mjcf", type=Path, default=DEFAULT_MJCF)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    try:
        filters = parse_filters(args.filters)
    except ValueError as e:
        p.error(str(e))

    if args.selftest:
        selftest()
        selftest_eef_slice()
        selftest_eef(args.eef_tool, args.eef_mjcf)
        selftest_deploy(args.vlahost_src)
        return
    # --checkpoint is the pretrained_model/ dir, not the job dir. Getting this wrong throws
    # a draccus ParsingError from deep inside config decoding, which says nothing useful.
    if args.checkpoint and not (args.checkpoint / "config.json").is_file():
        found = sorted(args.checkpoint.glob("run/checkpoints/*/pretrained_model"))
        hint = f"\n  did you mean:  --checkpoint {found[-1]}" if found else ""
        p.error(f"no config.json in {args.checkpoint} -- point --checkpoint at a "
                f"pretrained_model/ directory.{hint}")
    if args.dump_traces and args.dump_traces.is_dir():
        p.error(f"--dump-traces must be a file path, not a directory; "
                f"try {args.dump_traces / 'traces.npz'}")

    if not args.checkpoint or not args.dataset_root:
        p.error("--checkpoint and at least one --dataset-root are required")

    # cheap; never report numbers from a build whose accumulator or rewrite is broken
    selftest()
    selftest_deploy(args.vlahost_src)
    report = evaluate(
        checkpoint=args.checkpoint,
        dataset_roots=args.dataset_root,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_anchors_per_dataset=args.max_anchors_per_dataset,
        train_root=args.train_root,
        n_action_steps=args.n_action_steps,
        latency_steps=args.latency_steps,
        vlahost_src=args.vlahost_src,
        filters=filters,
        ablation=args.filter_ablation,
        invert_filter=args.keep_only_contaminated,
        dump_traces=args.dump_traces,
        trace_anchors=args.trace_anchors,
        trace_episodes=set(args.trace_episode) or None,
        seed=args.seed,
        seed_repeat=args.seed_repeat,
        eef_tool=args.eef_tool,
        eef_mjcf=args.eef_mjcf,
        eef=args.eef,
    )
    agg = report["aggregate"]
    print(f"\n=== aggregate over all held-out datasets "
          f"(executed horizon {report['executed_horizon']} steps) ===")
    raw = agg["policy_raw"]["mae"]
    null = agg["hold_state"]["mae"]
    for k, v in agg.items():
        delta = f"{100 * (v['mae'] - raw) / raw:+6.1f}% vs raw" if k.startswith("filt_") else ""
        print(f"  {k:<18} mae={v['mae']:.5f}  rmse={v['rmse']:.5f}  "
              f"norm_mae={v['norm_mae']:.5f}  tail={v['tail_ratio']:.2f}  "
              f"acc@0.25s={v.get('acc_at_tau', {}).get('0.25sigma', float('nan')):.3f}  {delta}")
    for k in ("policy_raw", "policy_deployed"):
        print(f"  {k} vs null: {null / max(agg[k]['mae'], 1e-12):.2f}x")
    eef_agg = report.get("eef_aggregate") or {}
    if "skipped" in eef_agg:
        print(f"\n  eef: skipped ({eef_agg['skipped']})")
    elif eef_agg:
        tight = next(iter(EEF_TAUS))
        label = f"{tight[0] * 1000:g}mm+{tight[1]:g}rad"
        print(f"\n=== end-effector pose error (mean over executed horizon) ===")
        for k, v in eef_agg.items():
            m, a = v["mean_per_channel"], v["acc_at_tau"][label]
            print(f"  {k:<18} "
                  f"L {m['left_pos_m'] * 1000:6.1f}mm/{m['left_rot_rad']:.3f}rad  "
                  f"R {m['right_pos_m'] * 1000:6.1f}mm/{m['right_rot_rad']:.3f}rad  "
                  f"acc@{label} L={a['left_pos_m']:.3f} R={a['right_pos_m']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
