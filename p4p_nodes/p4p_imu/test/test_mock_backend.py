"""Pure tests for the mock backend and the ImuBackend contract.

No ROS, no adafruit, no sensor. The mock is a first-class backend rather than a
test hook, so its contract is worth pinning: the node's fault handling is only as
trustworthy as the faults this can inject.
"""
import math

from p4p_imu import orientation
from p4p_imu.backends import ImuError, ImuResetError, make_backend
import pytest


class _Log:
    """Stand-in for a ROS logger, so nothing here needs rclpy."""

    def __init__(self):
        self.lines = []

    def info(self, msg, **kw):
        self.lines.append(('info', msg))

    def warning(self, msg, **kw):
        self.lines.append(('warning', msg))

    def error(self, msg, **kw):
        self.lines.append(('error', msg))


def _mock(**params):
    params.setdefault('sample_rate_hz', 100.0)
    return make_backend('mock', params, _Log())


def _started(**params):
    backend = _mock(**params)
    backend.start()
    return backend


def test_an_unknown_backend_is_refused_by_name():
    with pytest.raises(ImuError) as e:
        make_backend('bno055', {}, _Log())
    assert 'bno055' in str(e.value)
    assert 'mock' in str(e.value) and 'bno08x' in str(e.value)


def test_the_factory_never_substitutes_a_working_backend_for_a_broken_one():
    """Check that the factory raises for a misspelt backend instead of returning the mock."""
    with pytest.raises(ImuError):
        make_backend('bno08x_typo', {}, _Log())


def test_reading_before_start_is_an_error():
    with pytest.raises(ImuError):
        _mock().read()


def test_reading_after_stop_is_an_error():
    backend = _started()
    backend.read()
    backend.stop()
    with pytest.raises(ImuError):
        backend.read()


def test_stop_is_idempotent_and_never_raises():
    backend = _started()
    backend.stop()
    backend.stop()
    backend.stop()


def test_every_sample_is_valid():
    backend = _started()
    for _ in range(50):
        assert orientation.validate(backend.read()) is None


def test_consecutive_samples_differ():
    """Check that consecutive mock samples differ, so the staleness watchdog stays quiet."""
    backend = _started()
    first, second = backend.read(), backend.read()
    assert orientation.samples_differ(first, second)


def test_one_seed_gives_two_identical_streams():
    a = _started(mock_seed=1234)
    b = _started(mock_seed=1234)
    assert [a.read() for _ in range(20)] == [b.read() for _ in range(20)]


def test_different_seeds_give_different_streams():
    a = _started(mock_seed=1)
    b = _started(mock_seed=2)
    assert [a.read() for _ in range(20)] != [b.read() for _ in range(20)]


def test_the_gyro_carries_the_configured_turn_rate():
    """Check that gyro z carries the configured mock_yaw_rate."""
    backend = _started(mock_yaw_rate=0.75, mock_noise=0.0)
    assert backend.read().gyro[2] == pytest.approx(0.75)


def test_gravity_sits_in_plus_z_because_the_board_is_flat():
    backend = _started(mock_noise=0.0)
    assert backend.read().accel[2] == pytest.approx(9.80665)


def test_heading_advances_and_wraps_through_pi_without_unwrapping():
    """Check that the mock heading advances and wraps through pi without being unwrapped."""
    backend = _started(mock_yaw_rate=2.0, mock_noise=0.0)
    yaws = [orientation.yaw_from_quaternion(backend.read().quat) for _ in range(400)]
    assert all(-math.pi <= y <= math.pi for y in yaws)
    assert any(b - a < -math.pi for a, b in zip(yaws, yaws[1:]))


def test_a_spontaneous_reset_is_raised_once_then_heading_re_zeroes():
    """Check that mock_reset_after raises one reset, after which the heading re-zeroes."""
    backend = _started(mock_reset_after=5, mock_yaw_rate=2.0, mock_noise=0.0)
    for _ in range(5):
        backend.read()
    before = orientation.yaw_from_quaternion(backend._build().quat)
    assert abs(before) > 0.05

    with pytest.raises(ImuResetError):
        backend.read()

    after = orientation.yaw_from_quaternion(backend.read().quat)
    assert abs(after) < 0.05, 'heading should have re-zeroed, as the real chip does'

    for _ in range(10):
        backend.read()


def test_a_stall_returns_the_same_sample_forever_with_no_error():
    """Check that a stall serves the same sample forever without raising, like a silent reset."""
    backend = _started(mock_stall_after=3)
    for _ in range(3):
        backend.read()
    stuck = backend.read()
    for _ in range(20):
        assert not orientation.samples_differ(stuck, backend.read())


def test_a_refused_open_raises_until_it_is_allowed_to_succeed():
    backend = _mock(mock_fail_open=2)
    for _ in range(2):
        with pytest.raises(ImuError):
            backend.start()
    backend.start()
    assert orientation.validate(backend.read()) is None


def test_refused_opens_count_across_the_fresh_backends_the_node_makes():
    params = {'sample_rate_hz': 100.0, 'mock_fail_open': 2}
    for attempt in (1, 2):
        with pytest.raises(ImuError, match=f'refusing open {attempt}'):
            make_backend('mock', params, _Log()).start()
    backend = make_backend('mock', params, _Log())
    backend.start()
    assert orientation.validate(backend.read()) is None


def test_the_placeholder_window_serves_samples_the_validator_rejects():
    """Check that the placeholder window serves samples that fail validation, then real ones."""
    backend = _started(mock_placeholder_reads=3)
    for _ in range(3):
        assert orientation.validate(backend.read()) is not None
    assert orientation.validate(backend.read()) is None


def test_recover_before_start_is_an_error():
    with pytest.raises(ImuError):
        _mock().recover()
