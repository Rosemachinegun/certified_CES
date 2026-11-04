from argparse import Namespace
from unittest.mock import Mock

import numpy as np

import add.new_round as new_round_module
from add.new_round import (
    NEW_ROUND_HOME_ANGULAR_SPEED_DPS,
    NEW_ROUND_HOME_SPEED_MPS,
    return_grasp_hand_home_slowly,
)
from grasp_core.execution.config import HOME_CONFIG


def test_new_round_home_uses_slow_limits_and_preserves_start_pose(monkeypatch):
    publisher = Mock()
    start_pose = (np.array([0.65, -0.253, 0.87]), (0.0, 0.0, 0.0, 1.0))
    publish_home = Mock(return_value="right HOME")
    monkeypatch.setattr(new_round_module, "publish_home", publish_home)
    args = Namespace(
        target_trajectory_speed_mps=0.08,
        target_trajectory_angular_speed_dps=35.0,
    )

    status = return_grasp_hand_home_slowly(
        publisher, "right", args, start_pose=start_pose
    )

    assert status == "right HOME"
    call_args = publish_home.call_args
    assert call_args.args[:3] == (
        publisher,
        "right",
        HOME_CONFIG.position("right"),
    )
    runtime_args = call_args.args[3]
    assert runtime_args.target_trajectory_speed_mps == NEW_ROUND_HOME_SPEED_MPS
    assert (
        runtime_args.target_trajectory_angular_speed_dps
        == NEW_ROUND_HOME_ANGULAR_SPEED_DPS
    )
    assert call_args.kwargs["start_pose"] is start_pose
