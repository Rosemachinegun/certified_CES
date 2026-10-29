#!/usr/bin/env python3
"""Single-arm thank-you gesture used after a confirmed handover.

The grasping arm is deliberately never published here.  The other arm moves
from HOME to the chest, extends outward to thank the user, and returns HOME.
The right-arm gesture is the Y-axis mirror of the captured left-arm gesture.
"""

from __future__ import annotations

import argparse
from copy import copy

import numpy as np

from grasp_core.core.math.hand import normalize_hand
from grasp_core.core.math.pose import (
    PoseWaypoint,
    ik_home_wrist_orientation_quat,
    quaternion_angle_rad,
)
from grasp_core.execution.config import HOME_CONFIG
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    publish_path,
    read_current_tool_pose,
    request_ik_publisher_unavailable_status,
)
from grasp_core.execution.stop_control import publisher_stop_requested


# Captured left-tool translations in base_link, metres.
LEFT_CHEST_XYZ = (0.397, 0.079, 0.868)
LEFT_THANKS_XYZ = (0.470, 0.279, 0.851)

# The idle arm must really be at HOME before it is allowed to cross inward.
HOME_POSITION_TOLERANCE_M = 0.06
HOME_ORIENTATION_TOLERANCE_DEG = 12.0

# Slightly below the normal 0.08 m/s grasp trajectory.  Intermediate targets
# flow into one another without repeating a long deceleration/acceleration.
THANKS_LINEAR_SPEED_MPS = 0.075
THANKS_MIN_STEPS_PER_SEGMENT = 12


def gesture_hand_for_grasp_hand(grasp_hand: str) -> str:
    """Return the only hand that the thank-you action is allowed to command."""
    return "right" if normalize_hand(grasp_hand) == "left" else "left"


def thanks_xyz_for_hand(
    hand: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return chest/extension targets, mirrored for the right arm."""
    hand = normalize_hand(hand)
    chest_x, chest_y, chest_z = LEFT_CHEST_XYZ
    thanks_x, thanks_y, thanks_z = LEFT_THANKS_XYZ
    if hand == "left":
        return LEFT_CHEST_XYZ, LEFT_THANKS_XYZ
    return (
        (chest_x, -chest_y, chest_z),
        (thanks_x, -thanks_y, thanks_z),
    )


def execute_thanks(
    publisher: RequestIkTargetPublisher | None,
    grasp_hand: str,
    args: argparse.Namespace,
    *,
    start_pose: PoseWaypoint | None = None,
) -> str:
    """Run HOME -> chest -> thanks -> HOME using only the non-grasping arm."""
    grasp_hand = normalize_hand(grasp_hand)
    gesture_hand = gesture_hand_for_grasp_hand(grasp_hand)
    if publisher is None:
        return request_ik_publisher_unavailable_status(args)
    if publisher_stop_requested(publisher):
        return f"STOPPED before {gesture_hand} thank-you gesture"

    runtime_args = HOME_CONFIG.apply_runtime_args(copy(args), gesture_hand)
    runtime_args.target_trajectory_speed_mps = min(
        max(float(getattr(runtime_args, "target_trajectory_speed_mps", 0.08)), 1e-3),
        THANKS_LINEAR_SPEED_MPS,
    )
    home_position = np.asarray(HOME_CONFIG.position(gesture_hand), dtype=np.float64)
    home_orientation = ik_home_wrist_orientation_quat(
        runtime_args, hand=gesture_hand
    )
    if start_pose is None:
        start_position, start_orientation = read_current_tool_pose(
            publisher, gesture_hand
        )
    else:
        start_position = np.asarray(start_pose[0], dtype=np.float64)
        start_orientation = start_pose[1]

    position_error_m = float(np.linalg.norm(start_position - home_position))
    orientation_error_deg = float(np.rad2deg(
        quaternion_angle_rad(start_orientation, home_orientation)
    ))
    if position_error_m > HOME_POSITION_TOLERANCE_M:
        raise RuntimeError(
            f"thank-you refused: {gesture_hand} arm is not at HOME "
            f"(position error={position_error_m:.3f}m, "
            f"limit={HOME_POSITION_TOLERANCE_M:.3f}m)"
        )
    if orientation_error_deg > HOME_ORIENTATION_TOLERANCE_DEG:
        raise RuntimeError(
            f"thank-you refused: {gesture_hand} wrist is not at HOME "
            f"(orientation error={orientation_error_deg:.1f}deg, "
            f"limit={HOME_ORIENTATION_TOLERANCE_DEG:.1f}deg)"
        )

    chest_xyz, thanks_xyz = thanks_xyz_for_hand(gesture_hand)
    targets = (
        np.asarray(chest_xyz, dtype=np.float64),
        np.asarray(thanks_xyz, dtype=np.float64),
        home_position,
    )
    segment_start_position = start_position
    segment_start_orientation = start_orientation
    count = 0
    for index, target in enumerate(targets):
        segment_count = publish_path(
            publisher,
            gesture_hand,
            [(target, home_orientation)],
            runtime_args,
            start_position_xyz=segment_start_position,
            start_orientation_xyzw=segment_start_orientation,
            min_steps=THANKS_MIN_STEPS_PER_SEGMENT,
            final_hold_sec=(
                max(HOME_CONFIG.hold_sec, 0.0)
                if index == len(targets) - 1
                else 0.0
            ),
            startup_slowdown=index == 0,
            terminal_slowdown=index == len(targets) - 1,
        )
        count += segment_count
        if publisher_stop_requested(publisher):
            return f"STOPPED during {gesture_hand} thank-you gesture"
        if segment_count <= 0:
            raise RuntimeError(
                f"thank-you failed: {gesture_hand} segment {index + 1} "
                "published no pose target"
            )
        segment_start_position = target
        segment_start_orientation = home_orientation

    status = (
        f"{gesture_hand} thank-you completed; only {gesture_hand} targets published "
        f"(grasp hand={grasp_hand}); "
        f"chest=({chest_xyz[0]:.3f},{chest_xyz[1]:.3f},{chest_xyz[2]:.3f})m; "
        f"thanks=({thanks_xyz[0]:.3f},{thanks_xyz[1]:.3f},"
        f"{thanks_xyz[2]:.3f})m; returned HOME; count={count}"
    )
    print(f"[thanks] {status}", flush=True)
    return status
