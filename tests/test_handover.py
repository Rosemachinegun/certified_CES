from argparse import Namespace
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import numpy as np

import grasp_core.execution.skills.handover as handover_module
from grasp_core.execution.config import HOME_CONFIG
from grasp_core.execution.skills.handover import (
    HANDOVER_CONFIRM_TIMEOUT_SEC,
    LEFT_HANDOVER_XYZ,
    POST_RELEASE_HOLD_SEC,
    execute_handover_after_grasp,
    handover_xyz_for_hand,
)
from add.new_round import NEW_ROUND_REQUESTED_STATUS
from grasp_core.execution.result import FixedPlaceResult


def test_handover_tool_targets_are_mirrored() -> None:
    assert handover_xyz_for_hand("left") == LEFT_HANDOVER_XYZ
    x, y, z = LEFT_HANDOVER_XYZ
    assert handover_xyz_for_hand("right") == (x, -y, z)


def test_handover_waits_for_confirmation_then_releases_and_returns_home(
    monkeypatch,
) -> None:
    publisher = Mock()
    start_pose = (np.array([0.4, 0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    publish_path = Mock(return_value=12)
    release = Mock(return_value="Sent left gripper release")
    sleep = Mock()
    thanks_started = Event()
    thanks_finished = Event()
    home_started = Event()

    def run_thanks(*args, **kwargs):
        thanks_started.set()
        assert home_started.wait(timeout=1.0)
        thanks_finished.set()
        return "right thank-you completed"

    execute_thanks = Mock(side_effect=run_thanks)

    def run_home(*args, **kwargs):
        assert thanks_started.wait(timeout=1.0)
        home_started.set()
        return "Published left home target"

    publish_home = Mock(side_effect=run_home)
    def run_post_wave(*args, **kwargs):
        assert home_started.is_set()
        assert thanks_finished.is_set()
        return "Post-thanks wave completed"

    post_wave = Mock(side_effect=run_post_wave)
    monkeypatch.setattr(
        handover_module,
        "read_current_tool_pose",
        Mock(return_value=start_pose),
    )
    monkeypatch.setattr(
        handover_module,
        "publisher_stop_requested",
        Mock(return_value=False),
    )
    monkeypatch.setattr(handover_module, "publish_path", publish_path)
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(handover_module.time, "sleep", sleep)
    monkeypatch.setattr(handover_module, "execute_thanks", execute_thanks)
    monkeypatch.setattr(handover_module, "publish_home", publish_home)
    monkeypatch.setattr(
        handover_module, "execute_opening_wave_after_thanks", post_wave
    )
    release_requested = Event()
    ready = Event()
    result_holder = []

    worker = Thread(
        target=lambda: result_holder.append(
            execute_handover_after_grasp(
                publisher,
                "left",
                Namespace(),
                grasp_confirmed=True,
                release_requested=release_requested,
                ready_event=ready,
            )
        )
    )
    worker.start()

    assert ready.wait(timeout=1.0)
    release.assert_not_called()
    release_requested.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    result = result_holder[0]
    assert result.ok is True
    assert publish_path.call_args.kwargs["final_hold_sec"] == 0.0
    release.assert_called_once_with("release", ANY, hand="left")
    sleep.assert_called_once_with(POST_RELEASE_HOLD_SEC)
    execute_thanks.assert_called_once_with(
        publisher,
        "left",
        ANY,
        start_pose=start_pose,
    )
    publish_home.assert_called_once()
    home_args, home_kwargs = publish_home.call_args
    assert home_args[:3] == (
        publisher,
        "left",
        HOME_CONFIG.position("left"),
    )
    runtime_args = home_args[3]
    assert runtime_args.home_tilt_z_left_deg == HOME_CONFIG.tilt_z_left_deg
    assert runtime_args.home_tilt_y_left_deg == HOME_CONFIG.tilt_y_left_deg
    np.testing.assert_allclose(home_kwargs["start_pose"][0], LEFT_HANDOVER_XYZ)
    assert home_kwargs["start_pose"][1] == start_pose[1]
    assert home_kwargs["final_hold_sec"] == HOME_CONFIG.hold_sec
    post_wave.assert_called_once_with(publisher, "left")
    assert "Post-thanks wave completed" in result.status


def test_post_thanks_wave_runs_calibrated_wave_for_right_grasp(
    monkeypatch,
) -> None:
    publisher = Mock()
    execute_wave = Mock(return_value="Right-arm wave completed")
    monkeypatch.setitem(
        __import__("sys").modules,
        "add.wave",
        SimpleNamespace(execute_wave=execute_wave),
    )

    status = handover_module.execute_opening_wave_after_thanks(
        publisher, "right"
    )

    assert status == "Right-arm wave completed"
    execute_wave.assert_called_once_with(
        stop_requested=publisher.stop_requested
    )


def test_post_thanks_wave_does_not_misroute_left_grasp_to_right_arm() -> None:
    status = handover_module.execute_opening_wave_after_thanks(Mock(), "left")

    assert "skipped" in status
    assert "grasp hand was left" in status


def test_d_request_returns_home_then_opens_gripper(monkeypatch) -> None:
    publisher = Mock()
    start_pose = (np.array([0.4, 0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    events = []
    slow_home = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("home") or "Published left slow home target"
        )
    )
    release = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("release") or "Sent left gripper release"
        )
    )
    monkeypatch.setattr(
        handover_module, "read_current_tool_pose", Mock(return_value=start_pose)
    )
    monkeypatch.setattr(
        handover_module, "publisher_stop_requested", Mock(return_value=False)
    )
    monkeypatch.setattr(handover_module, "publish_path", Mock(return_value=12))
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(
        handover_module, "return_grasp_hand_home_slowly", slow_home
    )
    release_requested = Event()
    new_round_requested = Event()
    ready = Event()
    result_holder = []
    worker = Thread(
        target=lambda: result_holder.append(execute_handover_after_grasp(
            publisher,
            "left",
            Namespace(),
            grasp_confirmed=True,
            release_requested=release_requested,
            ready_event=ready,
            new_round_requested=new_round_requested,
        ))
    )
    worker.start()

    assert ready.wait(timeout=1.0)
    new_round_requested.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result_holder[0].ok is True
    assert NEW_ROUND_REQUESTED_STATUS in result_holder[0].status
    slow_home.assert_called_once()
    release.assert_called_once_with("release", ANY, hand="left")
    assert events == ["home", "release"]
    assert ready.is_set() is False


def test_handover_timeout_plays_prompt_then_uses_fixed_place(monkeypatch) -> None:
    assert HANDOVER_CONFIRM_TIMEOUT_SEC == 7.0
    publisher = Mock()
    start_pose = (np.array([0.4, 0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    events = []
    timeout_prompt = Mock(side_effect=lambda: events.append("prompt"))
    fallback_place = Mock(
        side_effect=lambda *args, **kwargs: (
            events.append("place") or FixedPlaceResult(True, "place release; HOME")
        )
    )
    release = Mock()
    monkeypatch.setattr(handover_module, "read_current_tool_pose", Mock(return_value=start_pose))
    monkeypatch.setattr(handover_module, "publisher_stop_requested", Mock(return_value=False))
    monkeypatch.setattr(handover_module, "publish_path", Mock(return_value=12))
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(handover_module, "execute_fixed_place_after_grasp", fallback_place)
    monkeypatch.setattr(handover_module, "HANDOVER_CONFIRM_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr(handover_module, "HANDOVER_CONFIRM_POLL_SEC", 0.0)
    ready = Event()

    result = execute_handover_after_grasp(
        publisher,
        "right",
        Namespace(),
        grasp_confirmed=True,
        release_requested=Event(),
        ready_event=ready,
        object_type="toy",
        timeout_prompt=timeout_prompt,
    )

    assert result.ok is True
    assert "timed out after 0s" in result.status
    assert events == ["prompt", "place"]
    assert ready.is_set() is False
    release.assert_not_called()
    fallback_place.assert_called_once_with(
        publisher,
        "right",
        ANY,
        grasp_confirmed=True,
        object_type="toy",
        target_count=1,
    )


def test_x_request_enters_handover_fallback_without_waiting_for_timeout(
    monkeypatch,
) -> None:
    publisher = Mock()
    start_pose = (np.array([0.4, -0.2, 0.7]), (0.0, 0.0, 0.0, 1.0))
    timeout_prompt = Mock()
    fallback_place = Mock(return_value=FixedPlaceResult(True, "place; HOME"))
    release = Mock()
    monkeypatch.setattr(
        handover_module, "read_current_tool_pose", Mock(return_value=start_pose)
    )
    monkeypatch.setattr(
        handover_module, "publisher_stop_requested", Mock(return_value=False)
    )
    monkeypatch.setattr(handover_module, "publish_path", Mock(return_value=12))
    monkeypatch.setattr(handover_module, "send_gripper_signal", release)
    monkeypatch.setattr(
        handover_module, "execute_fixed_place_after_grasp", fallback_place
    )
    monkeypatch.setattr(handover_module, "HANDOVER_CONFIRM_POLL_SEC", 0.0)
    fallback_requested = Event()
    fallback_requested.set()

    result = execute_handover_after_grasp(
        publisher,
        "right",
        Namespace(),
        grasp_confirmed=True,
        release_requested=Event(),
        ready_event=Event(),
        fallback_requested=fallback_requested,
        object_type="toy",
        timeout_prompt=timeout_prompt,
    )

    assert result.ok is True
    assert "requested by X" in result.status
    timeout_prompt.assert_called_once_with()
    release.assert_not_called()
    fallback_place.assert_called_once()
