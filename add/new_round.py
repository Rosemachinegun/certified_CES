"""Recovery helpers for an operator-detected missed grasp.

The D-key path is deliberately separate from automatic grip-failure recovery:
the robot may have reported a successful grasp, but a human/vision observer can
still see that the hand is empty before confirming the handover.
"""

from __future__ import annotations

import argparse
from copy import copy

from grasp_core.core.math.pose import PoseWaypoint
from grasp_core.execution.config import HOME_CONFIG
from grasp_core.execution.motion_executor import (
    RequestIkTargetPublisher,
    publish_home,
)


NEW_ROUND_SEQUENCE = ("07_1_try",)
START_NEW_ROUND = "start_new_round"
NEW_ROUND_REQUESTED_STATUS = "NEW_ROUND_REQUESTED"

# Intentionally slower than the normal 0.08 m/s / 35 deg/s grasp trajectory.
NEW_ROUND_HOME_SPEED_MPS = 0.04
NEW_ROUND_HOME_ANGULAR_SPEED_DPS = 20.0


def return_grasp_hand_home_slowly(
    publisher: RequestIkTargetPublisher,
    hand: str,
    args: argparse.Namespace,
    *,
    start_pose: PoseWaypoint,
) -> str:
    """Return the reported grasping hand HOME without opening its gripper."""
    home_args = copy(HOME_CONFIG.apply_runtime_args(args, hand))
    home_args.target_trajectory_speed_mps = min(
        float(getattr(
            home_args,
            "target_trajectory_speed_mps",
            NEW_ROUND_HOME_SPEED_MPS,
        )),
        NEW_ROUND_HOME_SPEED_MPS,
    )
    home_args.target_trajectory_angular_speed_dps = min(
        float(getattr(
            home_args,
            "target_trajectory_angular_speed_dps",
            NEW_ROUND_HOME_ANGULAR_SPEED_DPS,
        )),
        NEW_ROUND_HOME_ANGULAR_SPEED_DPS,
    )
    return publish_home(
        publisher,
        hand,
        HOME_CONFIG.position(hand),
        home_args,
        start_pose=start_pose,
        final_hold_sec=max(HOME_CONFIG.hold_sec, 0.0),
    )
