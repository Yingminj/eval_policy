# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for validating and repairing action-chunk trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SmallRollbackCorrection:
    """One short direction reversal removed from a joint trajectory."""

    joint_index: int
    start_index: int
    end_index: int
    rollback_magnitude: float


@dataclass(frozen=True)
class BoundaryRollbackCorrection:
    """A first-action rollback repaired relative to the real robot state."""

    joint_index: int
    join_index: int | None
    rollback_magnitude: float


@dataclass(frozen=True)
class OpenGripperLoopCorrection:
    """One joint-space loop removed while the corresponding gripper was open."""

    joint_index: int
    start_index: int
    extreme_index: int
    end_index: int
    excursion_magnitude: float
    return_gap: float
    max_abs_gripper: float


@dataclass(frozen=True)
class LargeExcursionCorrection:
    """A large internal peak or valley replaced by a monotonic trajectory."""

    joint_index: int
    extreme_type: str
    extreme_value: float
    deviation: float


def cubic_hermite_segment(
    start_value: float | torch.Tensor,
    end_value: float | torch.Tensor,
    num_steps: int,
    *,
    start_velocity: float | torch.Tensor = 0.0,
    end_velocity: float | torch.Tensor = 0.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a position- and velocity-continuous monotonic bridge.

    Velocities are expressed per action step.  Endpoint tangents are limited
    with the monotone cubic Hermite criterion so matching a noisy raw velocity
    cannot introduce an overshoot or a new rollback inside the bridge.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    start = torch.as_tensor(start_value, dtype=dtype, device=device)
    end = torch.as_tensor(end_value, dtype=start.dtype, device=start.device)
    if num_steps == 1:
        return start.unsqueeze(0)
    if num_steps == 2:
        return torch.stack([start, end])

    interval_count = num_steps - 1
    secant = (end - start) / interval_count
    secant_value = float(secant)
    if abs(secant_value) <= 1e-12:
        start_tangent = torch.zeros_like(start)
        end_tangent = torch.zeros_like(start)
    else:
        start_velocity_value = float(start_velocity)
        end_velocity_value = float(end_velocity)
        if start_velocity_value * secant_value <= 0:
            start_velocity_value = 0.0
        if end_velocity_value * secant_value <= 0:
            end_velocity_value = 0.0

        alpha = start_velocity_value / secant_value
        beta = end_velocity_value / secant_value
        tangent_norm = alpha * alpha + beta * beta
        if tangent_norm > 9.0:
            scale = 3.0 / math.sqrt(tangent_norm)
            start_velocity_value = scale * alpha * secant_value
            end_velocity_value = scale * beta * secant_value

        # Hermite uses derivatives with respect to normalized t, while the
        # caller supplies velocity per discrete action step.
        start_tangent = torch.as_tensor(
            start_velocity_value * interval_count,
            dtype=start.dtype,
            device=start.device,
        )
        end_tangent = torch.as_tensor(
            end_velocity_value * interval_count,
            dtype=start.dtype,
            device=start.device,
        )

    t = torch.linspace(0.0, 1.0, num_steps, dtype=start.dtype, device=start.device)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * start + h10 * start_tangent + h01 * end + h11 * end_tangent


def _nonzero_direction_runs(directions: list[int]) -> list[tuple[int, int, int]]:
    """Return ``(direction, start, end)`` runs, excluding stationary steps."""
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < len(directions):
        direction = directions[index]
        if direction == 0:
            index += 1
            continue

        end = index
        while end + 1 < len(directions) and directions[end + 1] == direction:
            end += 1
        runs.append((direction, index, end))
        index = end + 1
    return runs


def _direction_runs_ignoring_stationary(directions: list[int]) -> list[tuple[int, int, int]]:
    """Return direction runs while allowing stationary samples inside a run."""
    runs: list[list[int]] = []
    for index, direction in enumerate(directions):
        if direction == 0:
            continue
        if runs and runs[-1][0] == direction:
            runs[-1][2] = index
        else:
            runs.append([direction, index, index])
    return [(direction, start, end) for direction, start, end in runs]


def _centered_window(total_steps: int, start: int, end: int, window_size: int) -> tuple[int, int]:
    """Return a clamped ``[left, right)`` window around a delta run."""
    width = min(total_steps, window_size)
    center = (start + end) // 2
    left = max(0, center - width // 2)
    right = left + width
    if right > total_steps:
        right = total_steps
        left = right - width
    return left, right


def remove_small_rollbacks(
    chunk: torch.Tensor,
    *,
    joint_count: int = 14,
    window_size: int = 10,
    max_rollback_steps: int = 2,
    min_step_magnitude: float = 1e-3,
) -> tuple[torch.Tensor, list[SmallRollbackCorrection]]:
    """Remove isolated, short direction reversals from arm-joint trajectories.

    A rollback is repaired only when all of the following are true:

    - the opposite-direction run lasts at most ``max_rollback_steps``;
    - the non-stationary runs immediately before and after it have the same
      direction, so this is a local interruption rather than a real turn;
    - in the surrounding ``window_size`` deltas, the common direction clearly
      dominates (at least 4 supporting steps and at least 3:1 support).

    Long direction changes are deliberately left untouched so an intentional
    "reach and return" motion remains intact.  Corrected samples are replaced
    by a velocity-aware monotonic bridge between the normal samples on either
    side.  Rows are never deleted because doing so would desynchronise joints.

    Args:
        chunk: Action tensor shaped ``[time, action_dim]``.
        joint_count: Number of leading arm-joint columns to inspect.
        window_size: Number of adjacent action deltas used as local context.
        max_rollback_steps: Maximum consecutive reverse deltas to repair.
        min_step_magnitude: Deltas at or below this magnitude are stationary.

    Returns:
        A corrected clone of ``chunk`` and descriptions of every correction.
    """
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk with shape [time, action_dim], got {tuple(chunk.shape)}")
    if window_size < 3:
        raise ValueError(f"window_size must be >= 3, got {window_size}")
    if max_rollback_steps < 1:
        raise ValueError(f"max_rollback_steps must be >= 1, got {max_rollback_steps}")

    corrected = chunk.clone()
    corrections: list[SmallRollbackCorrection] = []
    if chunk.shape[0] < 4:
        return corrected, corrections

    inspected_joint_count = min(joint_count, chunk.shape[1])
    for joint_index in range(inspected_joint_count):
        original = chunk[:, joint_index]
        deltas = torch.diff(original)
        directions = torch.where(
            deltas > min_step_magnitude,
            1,
            torch.where(deltas < -min_step_magnitude, -1, 0),
        ).tolist()
        runs = _nonzero_direction_runs(directions)

        for run_index in range(1, len(runs) - 1):
            direction, delta_start, delta_end = runs[run_index]
            previous_direction = runs[run_index - 1][0]
            next_direction = runs[run_index + 1][0]
            rollback_steps = delta_end - delta_start + 1

            if rollback_steps > max_rollback_steps:
                continue
            if previous_direction != next_direction or direction == previous_direction:
                continue

            window_left, window_right = _centered_window(
                len(directions), delta_start, delta_end, window_size
            )
            window = directions[window_left:window_right]
            support_steps = sum(step == previous_direction for step in window)
            reverse_steps = sum(step == direction for step in window)
            if support_steps < 4 or support_steps < 3 * reverse_steps:
                continue
            if reverse_steps > max_rollback_steps:
                continue

            # A reverse delta at index i changes action[i] -> action[i + 1].
            # Use the end of the first normal delta after the rollback as the
            # right anchor, then replace only the samples inside both anchors.
            left_anchor = delta_start
            right_anchor = delta_end + 2
            if right_anchor >= chunk.shape[0]:
                continue

            start_velocity = (
                original[left_anchor] - original[left_anchor - 1]
                if left_anchor > 0
                else torch.zeros_like(original[left_anchor])
            )
            end_velocity = (
                original[right_anchor + 1] - original[right_anchor]
                if right_anchor + 1 < chunk.shape[0]
                else torch.zeros_like(original[right_anchor])
            )
            replacement = cubic_hermite_segment(
                original[left_anchor],
                original[right_anchor],
                right_anchor - left_anchor + 1,
                start_velocity=start_velocity,
                end_velocity=end_velocity,
                dtype=chunk.dtype,
                device=chunk.device,
            )
            corrected[left_anchor : right_anchor + 1, joint_index] = replacement

            rollback_values = original[left_anchor + 1 : delta_end + 2]
            rollback_magnitude = float(
                torch.max(torch.abs(rollback_values - original[left_anchor])).item()
            )
            corrections.append(
                SmallRollbackCorrection(
                    joint_index=joint_index,
                    start_index=left_anchor + 1,
                    end_index=delta_end + 1,
                    rollback_magnitude=rollback_magnitude,
                )
            )

    return corrected, corrections


def remove_boundary_rollbacks(
    chunk: torch.Tensor,
    current_positions: Sequence[float],
    *,
    joint_count: int = 14,
    window_size: int = 10,
    min_rollback_magnitude: float = math.radians(0.5),
    min_step_magnitude: float = 1e-3,
) -> tuple[torch.Tensor, list[BoundaryRollbackCorrection]]:
    """Remove a first-action rollback that opposes the chunk's early trend.

    The real robot position is treated as a virtual sample immediately before
    ``chunk[0]``.  A boundary is repaired only when its direction opposes a
    clearly dominant direction in the first ``window_size`` intra-chunk
    deltas.  The repair joins the current position to the first action that
    catches up with it in the dominant direction.  If the chunk never catches
    up, that joint holds its current position for this chunk.
    """
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk with shape [time, action_dim], got {tuple(chunk.shape)}")

    corrected = chunk.clone()
    corrections: list[BoundaryRollbackCorrection] = []
    inspected_joint_count = min(joint_count, chunk.shape[1], len(current_positions))
    if chunk.shape[0] < 2:
        return corrected, corrections

    for joint_index in range(inspected_joint_count):
        original = chunk[:, joint_index]
        current = float(current_positions[joint_index])
        boundary_delta = float(original[0]) - current
        if abs(boundary_delta) < min_rollback_magnitude:
            continue

        early_deltas = torch.diff(original[: min(chunk.shape[0], window_size + 1)])
        directions = torch.where(
            early_deltas > min_step_magnitude,
            1,
            torch.where(early_deltas < -min_step_magnitude, -1, 0),
        ).tolist()
        positive_steps = directions.count(1)
        negative_steps = directions.count(-1)
        if positive_steps == negative_steps:
            continue

        dominant_direction = 1 if positive_steps > negative_steps else -1
        support_steps = max(positive_steps, negative_steps)
        opposing_steps = min(positive_steps, negative_steps)
        if support_steps < 4 or support_steps < 3 * opposing_steps:
            continue
        if boundary_delta * dominant_direction >= 0:
            continue

        join_index: int | None = None
        for action_index, value in enumerate(original):
            if dominant_direction * (float(value) - current) >= -min_step_magnitude:
                join_index = action_index
                break

        if join_index is None:
            corrected[:, joint_index] = current
        else:
            end_velocity = (
                original[join_index + 1] - original[join_index]
                if join_index + 1 < chunk.shape[0]
                else torch.zeros_like(original[join_index])
            )
            corrected[: join_index + 1, joint_index] = cubic_hermite_segment(
                current,
                original[join_index],
                join_index + 1,
                start_velocity=0.0,
                end_velocity=end_velocity,
                dtype=chunk.dtype,
                device=chunk.device,
            )

        corrections.append(
            BoundaryRollbackCorrection(
                joint_index=joint_index,
                join_index=join_index,
                rollback_magnitude=abs(boundary_delta),
            )
        )

    return corrected, corrections


def remove_open_gripper_loops(
    chunk: torch.Tensor,
    *,
    joint_count: int = 14,
    joints_per_arm: int = 7,
    left_gripper_index: int = 14,
    right_gripper_index: int = 15,
    min_excursion: float = math.radians(1.0),
    max_excursion: float = math.radians(8.0),
    max_return_gap: float = math.radians(0.5),
    max_return_ratio: float = 0.2,
    max_duration_steps: int = 30,
    open_gripper_threshold: float = 0.1,
    gripper_margin_steps: int = 3,
    continuation_steps: int = 3,
    min_context_steps: int = 3,
    min_step_magnitude: float = 1e-3,
) -> tuple[torch.Tensor, list[OpenGripperLoopCorrection]]:
    """Remove small closed joint-space loops while the matching gripper is open.

    For each arm joint, a candidate has three parts: normal motion to point A,
    an opposite-direction excursion to B, and resumed normal motion to a point
    C close to A.  The A-to-C interval is replaced by a monotonic Hermite
    bridge only when the corresponding arm's gripper remains open throughout
    the interval (including a small temporal margin).

    The action rows are retained so arm and gripper timing stays synchronized.
    Large or long reach-and-return motions, grasping phases, and turns that do
    not return near their starting value are preserved.
    """
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk with shape [time, action_dim], got {tuple(chunk.shape)}")
    if joints_per_arm < 1:
        raise ValueError(f"joints_per_arm must be >= 1, got {joints_per_arm}")
    if min_excursion <= 0 or max_excursion < min_excursion:
        raise ValueError(
            f"Expected 0 < min_excursion <= max_excursion, got {min_excursion} and {max_excursion}"
        )
    if max_return_gap < 0 or not 0 <= max_return_ratio <= 1:
        raise ValueError(
            f"Invalid return limits: max_return_gap={max_return_gap}, "
            f"max_return_ratio={max_return_ratio}"
        )
    if max_duration_steps < 1:
        raise ValueError(f"max_duration_steps must be >= 1, got {max_duration_steps}")
    if open_gripper_threshold < 0 or gripper_margin_steps < 0:
        raise ValueError(
            f"Gripper thresholds must be non-negative, got {open_gripper_threshold} and "
            f"{gripper_margin_steps}"
        )
    if continuation_steps < 1 or min_context_steps < 1:
        raise ValueError(
            f"continuation_steps and min_context_steps must be >= 1, got "
            f"{continuation_steps} and {min_context_steps}"
        )

    inspected_joint_count = min(joint_count, chunk.shape[1])
    required_gripper_indices = {
        left_gripper_index if joint_index < joints_per_arm else right_gripper_index
        for joint_index in range(inspected_joint_count)
    }
    invalid_gripper_indices = [
        index for index in required_gripper_indices if index < 0 or index >= chunk.shape[1]
    ]
    if invalid_gripper_indices:
        raise ValueError(
            f"Gripper indices {sorted(invalid_gripper_indices)} are outside action width "
            f"{chunk.shape[1]}"
        )

    corrected = chunk.clone()
    corrections: list[OpenGripperLoopCorrection] = []
    if chunk.shape[0] < 2 * min_context_steps + 3:
        return corrected, corrections

    for joint_index in range(inspected_joint_count):
        original = chunk[:, joint_index]
        deltas = torch.diff(original)
        directions = torch.where(
            deltas > min_step_magnitude,
            1,
            torch.where(deltas < -min_step_magnitude, -1, 0),
        ).tolist()
        runs = _direction_runs_ignoring_stationary(directions)
        gripper_index = left_gripper_index if joint_index < joints_per_arm else right_gripper_index
        last_corrected_end = -1

        for run_index in range(1, len(runs) - 1):
            previous_direction, previous_start, previous_end = runs[run_index - 1]
            reverse_direction, reverse_start, reverse_end = runs[run_index]
            next_direction, _next_start, next_end = runs[run_index + 1]

            if previous_direction != next_direction or reverse_direction == previous_direction:
                continue
            previous_support = sum(
                direction == previous_direction
                for direction in directions[previous_start : previous_end + 1]
            )
            if previous_support < min_context_steps:
                continue

            start_index = reverse_start
            extreme_index = reverse_end + 1
            if start_index <= last_corrected_end:
                continue

            excursion_magnitude = abs(float(original[extreme_index] - original[start_index]))
            if not min_excursion <= excursion_magnitude <= max_excursion:
                continue

            latest_end_index = min(next_end + 1, start_index + max_duration_steps)
            if latest_end_index <= extreme_index:
                continue
            end_index = min(
                range(extreme_index + 1, latest_end_index + 1),
                key=lambda index: abs(float(original[index] - original[start_index])),
            )
            return_gap = abs(float(original[end_index] - original[start_index]))
            allowed_return_gap = min(max_return_gap, max_return_ratio * excursion_magnitude)
            if return_gap > allowed_return_gap:
                continue

            continued_steps = 0
            continuation_valid = True
            for direction in directions[end_index:]:
                if direction == 0:
                    continue
                if direction != previous_direction:
                    continuation_valid = False
                    break
                continued_steps += 1
                if continued_steps >= continuation_steps:
                    break
            if not continuation_valid or continued_steps < continuation_steps:
                continue

            gripper_start = max(0, start_index - gripper_margin_steps)
            gripper_end = min(chunk.shape[0], end_index + gripper_margin_steps + 1)
            max_abs_gripper = float(
                torch.max(torch.abs(chunk[gripper_start:gripper_end, gripper_index])).item()
            )
            if max_abs_gripper > open_gripper_threshold:
                continue

            start_velocity = (
                original[start_index] - original[start_index - 1]
                if start_index > 0
                else torch.zeros_like(original[start_index])
            )
            end_velocity = (
                original[end_index + 1] - original[end_index]
                if end_index + 1 < chunk.shape[0]
                else torch.zeros_like(original[end_index])
            )
            corrected[start_index : end_index + 1, joint_index] = cubic_hermite_segment(
                original[start_index],
                original[end_index],
                end_index - start_index + 1,
                start_velocity=start_velocity,
                end_velocity=end_velocity,
                dtype=chunk.dtype,
                device=chunk.device,
            )
            corrections.append(
                OpenGripperLoopCorrection(
                    joint_index=joint_index,
                    start_index=start_index,
                    extreme_index=extreme_index,
                    end_index=end_index,
                    excursion_magnitude=excursion_magnitude,
                    return_gap=return_gap,
                    max_abs_gripper=max_abs_gripper,
                )
            )
            last_corrected_end = end_index

    return corrected, corrections


def smooth_action_chunk(
    chunk: torch.Tensor,
    *,
    joint_count: int = 14,
    passes: int = 1,
) -> torch.Tensor:
    """Lightly smooth arm trajectories while preserving chunk endpoints.

    One binomial ``[0.25, 0.5, 0.25]`` pass reduces abrupt velocity changes
    without removing sustained reach-and-return motions.  The first and last
    action, and all non-arm columns such as grippers, remain unchanged.
    """
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk with shape [time, action_dim], got {tuple(chunk.shape)}")
    if passes < 0:
        raise ValueError(f"passes must be >= 0, got {passes}")
    if chunk.shape[0] < 3 or passes == 0:
        return chunk.clone()

    inspected_joint_count = min(joint_count, chunk.shape[1])
    smoothed = chunk.clone()
    for _ in range(passes):
        next_chunk = smoothed.clone()
        next_chunk[1:-1, :inspected_joint_count] = (
            0.25 * smoothed[:-2, :inspected_joint_count]
            + 0.5 * smoothed[1:-1, :inspected_joint_count]
            + 0.25 * smoothed[2:, :inspected_joint_count]
        )
        smoothed = next_chunk
    return smoothed


def smooth_large_excursions(
    chunk: torch.Tensor,
    *,
    joint_count: int = 14,
    wave_threshold: float = math.radians(100.0),
) -> tuple[torch.Tensor, list[LargeExcursionCorrection]]:
    """Replace very large internal peaks/valleys with monotonic trajectories.

    An extreme must be farther than ``wave_threshold`` from *both* endpoints.
    This matches the original intent while avoiding false positives on a large
    monotonic move whose extreme is itself the final action.
    """
    if chunk.ndim != 2:
        raise ValueError(f"Expected chunk with shape [time, action_dim], got {tuple(chunk.shape)}")

    corrected = chunk.clone()
    corrections: list[LargeExcursionCorrection] = []
    if chunk.shape[0] < 3:
        return corrected, corrections

    inspected_joint_count = min(joint_count, chunk.shape[1])
    for joint_index in range(inspected_joint_count):
        trajectory = chunk[:, joint_index]
        start_value = float(trajectory[0])
        end_value = float(trajectory[-1])
        max_value = float(trajectory.max())
        min_value = float(trajectory.min())

        peak_deviation = min(abs(max_value - start_value), abs(max_value - end_value))
        valley_deviation = min(abs(min_value - start_value), abs(min_value - end_value))

        if peak_deviation > wave_threshold and peak_deviation >= valley_deviation:
            extreme_type = "peak"
            extreme_value = max_value
            deviation = peak_deviation
        elif valley_deviation > wave_threshold:
            extreme_type = "valley"
            extreme_value = min_value
            deviation = valley_deviation
        else:
            continue

        corrected[:, joint_index] = torch.linspace(
            start_value,
            extreme_value,
            chunk.shape[0],
            dtype=chunk.dtype,
            device=chunk.device,
        )
        corrections.append(
            LargeExcursionCorrection(
                joint_index=joint_index,
                extreme_type=extreme_type,
                extreme_value=extreme_value,
                deviation=deviation,
            )
        )

    return corrected, corrections
