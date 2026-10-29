from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import add.thanks as thanks_module
from add.thanks import (
    LEFT_CHEST_XYZ,
    LEFT_THANKS_XYZ,
    execute_thanks,
    gesture_hand_for_grasp_hand,
    thanks_xyz_for_hand,
)


def test_thanks_targets_are_mirrored() -> None:
    assert thanks_xyz_for_hand("left") == (LEFT_CHEST_XYZ, LEFT_THANKS_XYZ)
    chest, extended = thanks_xyz_for_hand("right")
    assert chest == (LEFT_CHEST_XYZ[0], -LEFT_CHEST_XYZ[1], LEFT_CHEST_XYZ[2])
    assert extended == (
        LEFT_THANKS_XYZ[0], -LEFT_THANKS_XYZ[1], LEFT_THANKS_XYZ[2]
    )


def test_thanks_uses_only_opposite_hand_and_returns_home(monkeypatch) -> None:
    assert gesture_hand_for_grasp_hand("right") == "left"
    publisher = Mock()
    publisher.stop_requested.return_value = False
    args = SimpleNamespace(target_trajectory_speed_mps=0.08)
    home_orientation = (0.0, 0.0, 0.0, 1.0)
    start = np.asarray(thanks_module.HOME_CONFIG.position("left"), dtype=np.float64)
    monkeypatch.setattr(
        thanks_module, "ik_home_wrist_orientation_quat",
        Mock(return_value=home_orientation),
    )
    monkeypatch.setattr(
        thanks_module, "read_current_tool_pose",
        Mock(return_value=(start, home_orientation)),
    )
    publish_path = Mock(return_value=123)
    monkeypatch.setattr(thanks_module, "publish_path", publish_path)

    status = execute_thanks(publisher, "right", args)

    assert publish_path.call_count == 3
    calls = publish_path.call_args_list
    assert all(call.args[1] == "left" for call in calls)
    assert tuple(calls[0].args[2][0][0]) == LEFT_CHEST_XYZ
    assert tuple(calls[1].args[2][0][0]) == LEFT_THANKS_XYZ
    assert tuple(calls[2].args[2][0][0]) == thanks_module.HOME_CONFIG.position("left")
    assert calls[0].kwargs["start_position_xyz"] is start
    assert calls[1].kwargs["start_position_xyz"] is calls[0].args[2][0][0]
    assert calls[2].kwargs["start_position_xyz"] is calls[1].args[2][0][0]
    assert calls[0].args[3].target_trajectory_speed_mps == 0.075
    assert calls[0].kwargs["startup_slowdown"] is True
    assert calls[0].kwargs["terminal_slowdown"] is False
    assert calls[1].kwargs["startup_slowdown"] is False
    assert calls[1].kwargs["terminal_slowdown"] is False
    assert calls[2].kwargs["startup_slowdown"] is False
    assert calls[2].kwargs["terminal_slowdown"] is True
    assert "only left targets published (grasp hand=right)" in status


def test_thanks_refuses_if_idle_arm_is_not_home(monkeypatch) -> None:
    publisher = Mock()
    publisher.stop_requested.return_value = False
    args = SimpleNamespace(target_trajectory_speed_mps=0.08)
    orientation = (0.0, 0.0, 0.0, 1.0)
    monkeypatch.setattr(
        thanks_module, "ik_home_wrist_orientation_quat", Mock(return_value=orientation)
    )
    monkeypatch.setattr(
        thanks_module,
        "read_current_tool_pose",
        Mock(return_value=(np.asarray((0.6, 0.6, 0.6)), orientation)),
    )
    publish_path = Mock()
    monkeypatch.setattr(thanks_module, "publish_path", publish_path)

    try:
        execute_thanks(publisher, "right", args)
    except RuntimeError as exc:
        assert "is not at HOME" in str(exc)
    else:
        raise AssertionError("unsafe non-HOME gesture start was accepted")
    publish_path.assert_not_called()
