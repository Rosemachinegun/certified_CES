"""Exercise the wave entry gate with simulated ROS feedback, without hardware."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def wave(monkeypatch):
    ros = SimpleNamespace(ok=lambda: True)
    monkeypatch.setitem(sys.modules, 'rclpy', ros)
    monkeypatch.setitem(sys.modules, 'rclpy.node', SimpleNamespace(Node=object))
    monkeypatch.setitem(sys.modules, 'rclpy.qos', SimpleNamespace(
        DurabilityPolicy=object, HistoryPolicy=object, QoSProfile=object,
        ReliabilityPolicy=object, qos_profile_sensor_data=None,
    ))
    monkeypatch.setitem(sys.modules, 'std_msgs.msg', SimpleNamespace(Int32=object))
    monkeypatch.setitem(sys.modules, 'grasp_core.communication.marvin_interfaces',
                        SimpleNamespace(load_marvin_interfaces=lambda: (object,) * 3))
    spec = importlib.util.spec_from_file_location(
        'wave_under_test', Path(__file__).resolve().parents[1] / 'add/wave.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_gate(wave, monkeypatch, samples):
    node = wave.WaveGreeting.__new__(wave.WaveGreeting)
    node.feedback = None
    node.feedback_sequence = 0
    node.stop_requested = lambda: False
    node.required_input_mode = None
    node.get_logger = Mock(return_value=Mock())
    elapsed = [0.0]
    index = [0]

    def spin(*args, **kwargs):
        elapsed[0] += 0.05
        positions, velocities = samples[min(index[0], len(samples) - 1)]
        node.feedback = SimpleNamespace(arm_positions=positions, arm_velocities=velocities)
        node.feedback_sequence += 1
        index[0] += 1

    monkeypatch.setattr(wave, 'time', SimpleNamespace(monotonic=lambda: elapsed[0]))
    wave.rclpy.spin_once = spin
    result = node.wait_for_safe_home_start(timeout=1.0)
    return result, elapsed[0]


def sample(wave, joint=5, delta=0.0, speed=0.0):
    positions = [0.0] * 7 + list(wave.P0_HOME)
    positions[7 + joint] += delta
    return positions, [speed] * 14


@pytest.mark.parametrize('delta', [0.0, 0.308, -0.308])
def test_measured_home_is_accepted_and_preserved(wave, monkeypatch, delta):
    positions, velocities = sample(wave, delta=delta)
    result, elapsed = run_gate(wave, monkeypatch, [(positions, velocities)])
    assert result == (positions[:7], positions[7:])
    assert elapsed >= wave.STATIONARY_DURATION_SEC


@pytest.mark.parametrize('joint,delta', [(5, 0.36), (0, 0.13), (4, 0.13), (6, 0.13)])
def test_outside_home_still_rejected(wave, monkeypatch, joint, delta):
    with pytest.raises(RuntimeError, match=f'joint {joint + 1} error='):
        run_gate(wave, monkeypatch, [sample(wave, joint=joint, delta=delta)])


def test_home_can_converge_during_wait(wave, monkeypatch):
    good = sample(wave, delta=0.308)
    result, elapsed = run_gate(wave, monkeypatch, [sample(wave, delta=0.4)] * 4 + [good])
    assert result[1] == good[0][7:]
    assert elapsed >= 0.4


def test_moving_arms_still_rejected(wave, monkeypatch):
    with pytest.raises(RuntimeError, match='did not remain stationary'):
        run_gate(wave, monkeypatch, [sample(wave, delta=0.308, speed=0.03)])


@pytest.mark.parametrize('invalid', ['nan', 'short'])
def test_invalid_feedback_rejected(wave, monkeypatch, invalid):
    positions, velocities = sample(wave)
    if invalid == 'nan':
        positions[12] = float('nan')
    else:
        velocities.pop()
    with pytest.raises(RuntimeError, match='non-finite|expected 14'):
        run_gate(wave, monkeypatch, [(positions, velocities)])
