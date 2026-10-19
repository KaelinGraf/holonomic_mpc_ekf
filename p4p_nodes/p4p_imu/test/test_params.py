"""Parameter validation: what is clamped, what is fatal, and what is derived.

The split matters. A rate out of range is a field condition and the node clamps
it loudly, because a bot already on the floor is better off running. A bad
enumeration is a typo, and running the wrong backend silently is worse than
refusing to start.
"""
import pytest


def test_a_rate_below_the_floor_is_clamped_rather_than_fatal(make_rig):
    rig = make_rig(sample_rate_hz=0.1)
    assert rig.node._sample_rate == pytest.approx(1.0)


def test_an_absurd_rate_is_capped(make_rig):
    """Check that a sample rate above 400 Hz is capped at 400 Hz."""
    rig = make_rig(sample_rate_hz=100000.0)
    assert rig.node._sample_rate == pytest.approx(400.0)


def test_the_report_period_follows_the_clamped_rate(make_rig):
    """Check that the report period and the backend's rate follow the clamped sample rate."""
    rig = make_rig(sample_rate_hz=0.1)
    assert rig.node._report_period == pytest.approx(1.0)
    assert rig.node._params['sample_rate_hz'] == pytest.approx(1.0)


def test_we_poll_faster_than_the_sensor_reports(make_rig):
    """Check that the poll period is the report period divided by the oversample factor."""
    rig = make_rig(sample_rate_hz=200.0, oversample=2.0)
    assert rig.node._report_period == pytest.approx(1.0 / 200.0)
    assert rig.node._poll_period == pytest.approx(1.0 / 400.0)


def test_oversampling_below_one_is_clamped(make_rig):
    """Check that an oversample factor below one is clamped to one."""
    rig = make_rig(sample_rate_hz=100.0, oversample=0.25)
    assert rig.node._oversample == pytest.approx(1.0)
    assert rig.node._poll_period == pytest.approx(rig.node._report_period)


def test_the_staleness_timeout_is_floored_on_report_periods_not_poll_periods(make_rig):
    """Check that data_timeout is floored at two report periods, not two poll periods."""
    rig = make_rig(sample_rate_hz=10.0, oversample=4.0, data_timeout=0.001)
    assert rig.node._data_timeout >= 2.0 / 10.0


def test_the_queue_holds_a_fixed_time_rather_than_a_fixed_sample_count(make_rig):
    """Check that the publisher queue depth scales with the sample rate to hold 0.1 s."""
    assert make_rig(sample_rate_hz=200.0).node._pub.qos_profile.depth == 20
    assert make_rig(sample_rate_hz=400.0).node._pub.qos_profile.depth == 40


def test_the_queue_never_drops_below_ten(make_rig):
    """Check that the publisher queue depth never drops below ten at low rates."""
    assert make_rig(sample_rate_hz=20.0).node._pub.qos_profile.depth == 10


def test_an_unknown_backend_refuses_to_start(make_rig):
    """Check that an unknown imu_backend raises ValueError instead of falling back."""
    with pytest.raises(ValueError) as e:
        make_rig(imu_backend='bno085_typo')
    assert 'bno085_typo' in str(e.value)


def test_the_i2c_address_reaches_the_backend(make_rig):
    """Check that a non-default I2C address such as 0x4B reaches the backend parameters."""
    rig = make_rig(i2c_address=0x4B)
    assert rig.node._params['i2c_address'] == 0x4B


def test_the_bus_number_reaches_the_backend(make_rig):
    """Check that a non-default I2C bus number reaches the backend parameters."""
    rig = make_rig(i2c_bus=3)
    assert rig.node._params['i2c_bus'] == 3
