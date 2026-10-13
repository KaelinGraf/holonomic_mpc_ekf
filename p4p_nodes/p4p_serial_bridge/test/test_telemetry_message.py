"""serial_bridge.fill_telemetry against the DriveTelemetry that is installed.

    python3 -m pytest p4p_nodes/p4p_serial_bridge/test/test_telemetry_message.py -q

The firmware and convchart_interfaces gained the request and pulse columns on the
same day, and a deploy updates them one at a time: rows of either firmware must
fill a message of either age without raising, since a raise in the read timer
would take the bridge down with the robot on the floor. Which half runs depends
on the convchart_interfaces sourced. Needs ROS and the bridge's own imports;
skipped otherwise.
"""
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('serial')
DriveTelemetry = pytest.importorskip('convchart_interfaces.msg').DriveTelemetry

from p4p_serial_bridge import protocol  # noqa: E402, I100
from p4p_serial_bridge.serial_bridge import fill_telemetry, TELEMETRY_HAS_PULSES  # noqa: E402

NEW = '1516,0.0000,0.1,0.200,0.000,-0.300,1,0,2,0.200,0.000,-0.300,1140,1220,1780,1860'
OLD = '1516,0.0000,0.1,0.200,0.000,-0.300,1,0,2'


@pytest.mark.parametrize('row', [NEW, OLD])
def test_either_row_fills_the_installed_message(row):
    """Both row formats fill whichever message is installed, the shared fields alike."""
    msg = DriveTelemetry()
    fill_telemetry(msg, protocol.parse_line(row))
    assert (msg.mega_t_ms, msg.flags, msg.bad_commands) == (1516, 1, 2)
    assert msg.armed and not msg.timed_out and not msg.saturated
    assert (msg.applied.linear.x, msg.applied.angular.z) == pytest.approx((0.2, -0.3))


@pytest.mark.skipif(not TELEMETRY_HAS_PULSES, reason='this convchart_interfaces predates pulse_us')
def test_the_new_columns_reach_the_message():
    """With both ends updated, the request and the four pulses are published."""
    msg = DriveTelemetry()
    fill_telemetry(msg, protocol.parse_line(NEW))
    assert (msg.requested.linear.x, msg.requested.linear.y,
            msg.requested.angular.z) == pytest.approx((0.2, 0.0, -0.3))
    assert list(msg.pulse_us) == [1140, 1220, 1780, 1860]


@pytest.mark.skipif(not TELEMETRY_HAS_PULSES, reason='this convchart_interfaces predates pulse_us')
def test_older_firmware_leaves_the_new_fields_zero():
    """All-zero pulses are how a reader tells 'not reported' from neutral (1500)."""
    msg = DriveTelemetry()
    fill_telemetry(msg, protocol.parse_line(OLD))
    assert list(msg.pulse_us) == [0, 0, 0, 0]
    assert msg.requested.linear.x == 0.0
