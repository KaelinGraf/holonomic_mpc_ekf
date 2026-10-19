"""Round-trip the bridge's wire format against the firmware simulator.

Both ends are pure Python, so this runs with no ROS, no Docker, no serial port
and no Arduino:

    python3 -m pytest p4p_nodes/p4p_serial_bridge/test/test_serial_protocol.py -q

p4p_serial_bridge.protocol is what the node puts on the wire;
tools.fake_mega is an independent reimplementation of what the firmware expects.
A bug in either one shows up here as a disagreement.
"""
import math
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parents[1] / 'tools'))
sys.path.insert(0, str(PACKAGE / 'p4p_serial_bridge'))

import fake_mega  # noqa: E402, I100
import protocol  # noqa: E402, I100


def drive(mega, t_from, t_to, feed=None, at=None):
    """Pump the simulator over a span of virtual time and return the rows it decoded to.

    Args:
        mega: FakeMega simulator.
        t_from: First virtual time to pump [ms].
        t_to: Last virtual time to pump, inclusive [ms].
        feed: Bytes to feed the simulator once, or None.
        at: Virtual time to feed them at [ms]; None feeds them on the first step.
    """
    rows = []
    for t in range(t_from, t_to + 1):
        if feed is not None and (at is None or t == at):
            mega.feed(feed)
            feed = None
        mega.pump(t)
        for line in mega.drain().decode().split('\r\n'):
            tel = protocol.parse_line(line)
            if tel is not None:
                rows.append(tel)
    return rows


@pytest.fixture
def mega():
    m = fake_mega.FakeMega()
    m.drain()
    return m


def test_constants_match_the_firmware():
    """The duplicated firmware constants in both modules must agree."""
    assert protocol.TELEMETRY_HEADER == fake_mega.TELEMETRY_HEADER
    assert protocol.LEGACY_TELEMETRY_HEADER == fake_mega.LEGACY_TELEMETRY_HEADER
    assert protocol.FIELDS == len(protocol.TELEMETRY_HEADER.split(','))
    assert protocol.LEGACY_FIELDS == len(protocol.LEGACY_TELEMETRY_HEADER.split(','))
    assert protocol.FLAG_ARMED == fake_mega.FLAG_ARMED
    assert protocol.FLAG_TIMEOUT == fake_mega.FLAG_TIMEOUT
    assert protocol.FLAG_SATURATED == fake_mega.FLAG_SATURATED
    assert protocol.FIRMWARE_CMD_TIMEOUT_S * 1000 == fake_mega.MegaConfig.cmd_timeout_ms
    assert protocol.MAX_LINE_LEN == fake_mega.MegaConfig.cmd_buf_len - 1


@pytest.mark.parametrize('legacy', [False, True])
def test_banner_and_header_are_not_data(legacy):
    """The boot banner's last line is a bare CSV header, and must not parse."""
    fresh = fake_mega.FakeMega(legacy_telemetry=legacy)
    lines = fresh.drain().decode().split('\r\n')
    header = protocol.LEGACY_TELEMETRY_HEADER if legacy else protocol.TELEMETRY_HEADER
    assert header in lines
    assert all(protocol.parse_line(ln) is None for ln in lines)
    assert all(protocol.is_header(ln) == ln.startswith('t_ms,') for ln in lines)


def test_arm_then_drive_round_trips(mega):
    """A formatted V line is accepted, acted on, and echoed back unchanged."""
    mega.feed(protocol.ARM + protocol.format_velocity(0.25, -0.1, 0.4))
    rows = drive(mega, 0, 60)
    assert rows, 'no telemetry decoded'
    last = rows[-1]
    assert last.armed and not last.timed_out
    assert last.applied == pytest.approx((0.25, -0.1, 0.4), abs=1e-3)
    assert mega.bad == 0, 'the firmware rejected a line we formatted'


def test_rows_carry_the_request_as_received_and_the_pulses(mega):
    """The extra columns: the V the Mega took, and what it wrote to the motors."""
    mega.feed(protocol.ARM + protocol.format_velocity(0.0, 0.1, -0.3))
    last = drive(mega, 0, 60)[-1]
    assert last.requested == pytest.approx((0.0, 0.1, -0.3), abs=1e-3)
    assert last.pulse_us == mega.pulses
    assert all(us != 1500 for us in last.pulse_us), 'every wheel turns for a strafe + turn'


def test_disarmed_rows_report_neutral_pulses(mega):
    """Before E nothing is driven: the request is zero and all four pulses neutral."""
    last = drive(mega, 0, 60)[-1]
    assert not last.armed
    assert last.requested == (0.0, 0.0, 0.0)
    assert last.pulse_us == (1500, 1500, 1500, 1500)


def test_legacy_firmware_rows_still_parse():
    """Legacy nine-column rows decode, with no request or pulse widths."""
    old = fake_mega.FakeMega(legacy_telemetry=True)
    old.drain()
    old.feed(protocol.ARM + protocol.format_velocity(0.25, 0.0, 0.0))
    rows = drive(old, 0, 60)
    assert rows, 'no telemetry decoded'
    assert rows[-1].applied == pytest.approx((0.25, 0.0, 0.0), abs=1e-3)
    assert all(r.requested is None and r.pulse_us is None for r in rows)


def test_the_firmware_accepts_every_formatted_velocity(mega):
    """Whatever the node clamps to, the formatter must produce a legal line."""
    mega.feed(protocol.ARM)
    for vx, vy, wz in ((0.0, 0.0, 0.0), (0.5, 0.5, 3.02), (-0.5, -0.5, -3.02),
                       (0.001, -0.001, 0.0005), (1e-7, 0.0, 0.0)):
        mega.feed(protocol.format_velocity(vx, vy, wz))
    drive(mega, 0, 60)
    assert mega.bad == 0


def test_clamping_cannot_launder_a_non_finite_command():
    """Clamping must never turn NaN or inf into a legal full-scale command."""
    def clamp(v, limit):
        return max(-limit, min(limit, v))

    for bad in (float('nan'), float('inf'), -float('inf')):
        laundered = clamp(bad, 0.5)
        assert math.isfinite(laundered), 'precondition of the trap being tested'
        assert not math.isfinite(bad)
    m = fake_mega.FakeMega()
    m.drain()
    m.feed(b'E\nV,nan,0,0\nV,inf,0,0\n')
    drive(m, 0, 40)
    assert m.bad == 2, 'the firmware must reject non-finite V fields'


def test_non_finite_and_over_long_are_refused():
    """The formatter refuses rather than letting the firmware count an error."""
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            protocol.format_velocity(bad, 0.0, 0.0)
    with pytest.raises(ValueError):
        protocol.format_velocity(1e12, -1e12, 1e12)


def test_our_rate_beats_the_firmware_watchdog(mega):
    """Sending at command_rate_hz must never let TIMEOUT appear."""
    period_ms = int(1000 / 20.0)
    assert period_ms < fake_mega.MegaConfig.cmd_timeout_ms
    mega.feed(protocol.ARM)
    rows = []
    for t in range(0, 2000):
        if t % period_ms == 0:
            mega.feed(protocol.format_velocity(0.2, 0.0, 0.0))
        mega.pump(t)
        rows += [r for r in (protocol.parse_line(ln)
                             for ln in mega.drain().decode().split('\r\n')) if r]
    assert rows
    assert not any(r.timed_out for r in rows), 'watchdog fired at the default rate'
    assert all(r.armed for r in rows)


def test_silence_trips_the_watchdog_but_stays_armed(mega):
    """Stopping mid-drive zeroes the echo and sets TIMEOUT, without disarming."""
    mega.feed(protocol.ARM + protocol.format_velocity(0.3, 0.0, 0.0))
    moving = drive(mega, 0, 100)
    assert moving[-1].applied[0] == pytest.approx(0.3, abs=1e-3)
    stopped = drive(mega, 101, 700)
    assert stopped[-1].timed_out
    assert stopped[-1].armed, 'the firmware must stay armed across a timeout'
    assert stopped[-1].applied == (0.0, 0.0, 0.0)
    assert stopped[-1].requested == pytest.approx((0.3, 0.0, 0.0), abs=1e-3)
    assert stopped[-1].pulse_us == (1500, 1500, 1500, 1500)


def test_disarm_is_immediate(mega):
    """S zeroes the echo and clears ARMED on the next control cycle."""
    mega.feed(protocol.ARM + protocol.format_velocity(0.3, 0.0, 0.0))
    drive(mega, 0, 100)
    mega.feed(protocol.DISARM)
    rows = drive(mega, 101, 160)
    assert not rows[-1].armed
    assert rows[-1].applied == (0.0, 0.0, 0.0)


def test_saturation_is_visible_in_the_echo(mega):
    """An over-range command comes back scaled, with SAT set."""
    mega.feed(protocol.ARM + protocol.format_velocity(0.5, 0.0, 2.0))
    rows = drive(mega, 0, 60)
    last = rows[-1]
    assert last.saturated
    assert last.applied[0] < 0.5 and last.applied[2] < 2.0
    assert last.applied[2] / last.applied[0] == pytest.approx(2.0 / 0.5, rel=1e-3)
    assert last.requested == pytest.approx((0.5, 0.0, 2.0), abs=1e-3)
    assert max(abs(us - 1500) for us in last.pulse_us) == 800


def test_line_reader_reassembles_split_chunks():
    """Telemetry split anywhere, including between CR and LF, must decode."""
    source = fake_mega.FakeMega()
    source.feed(protocol.ARM + protocol.format_velocity(0.1, 0.0, 0.0))
    blob = b''
    for t in range(0, 200):
        source.pump(t)
        blob += source.drain()
    whole = [r for r in (protocol.parse_line(ln)
                         for ln in blob.decode().split('\r\n')) if r]

    reader = protocol.LineReader()
    piecewise = []
    for i in range(0, len(blob), 7):
        for line in reader.feed(blob[i:i + 7]):
            tel = protocol.parse_line(line)
            if tel is not None:
                piecewise.append(tel)
    assert piecewise == whole
    assert reader.overruns == 0


def test_line_reader_bounds_a_stream_with_no_newlines():
    """Line noise must not grow the buffer without bound."""
    reader = protocol.LineReader(max_len=64)
    for _ in range(50):
        assert reader.feed(b'x' * 32) == []
    assert reader.overruns > 0


def test_line_reader_keeps_every_row_in_an_oversized_read():
    """A read larger than max_len keeps every complete row."""
    source = fake_mega.FakeMega()
    source.feed(protocol.ARM + protocol.format_velocity(0.1, 0.0, 0.0))
    blob = b''
    for t in range(0, 500):
        source.pump(t)
        blob += source.drain()
    expected = len([r for r in (protocol.parse_line(ln)
                                for ln in blob.decode().split('\r\n')) if r])
    assert expected > 20, 'need a decent number of rows to make this meaningful'

    reader = protocol.LineReader(max_len=512)
    got = []
    for i in range(0, len(blob), 1100):
        for line in reader.feed(blob[i:i + 1100]):
            tel = protocol.parse_line(line)
            if tel is not None:
                got.append(tel)
    assert len(got) == expected, f'lost {expected - len(got)} of {expected} rows'
    assert reader.overruns == 0, 'a stream full of newlines must not count overruns'


def test_unparseable_rows_are_rejected_not_guessed():
    """Short rows, junk and the firmware's own nan/inf/ovf output return None."""
    good = '100,0.0,0.0,0,0,0,1,0,0,0.1,0,0,1500,1500,1500,1500'
    assert protocol.parse_line(good) is not None, 'the template itself must parse'
    for line in ('', '   ', '# status', protocol.TELEMETRY_HEADER,
                 protocol.LEGACY_TELEMETRY_HEADER, 't_ms,some,future,header',
                 '1,2,3', '1,2,3,4,5,6,7,8,9,10', 'a,b,c,d,e,f,g,h,i',
                 '100,nan,0.0,0,0,0,1,0,0', '100,inf,0.0,0,0,0,1,0,0',
                 '100,ovf,0.0,0,0,0,1,0,0',
                 good + ',1500', good.rsplit(',', 1)[0],
                 good.replace('0.1,', 'nan,'), good.replace('0.1,', 'ovf,'),
                 good.replace(',1500,1500,1500,1500', ',1500.0,1500,1500,1500'),
                 good.replace(',1500,1500,1500,1500', ',1500,-1,1500,1500'),
                 good.replace(',1500,1500,1500,1500', ',1500,1500,65536,1500')):
        assert protocol.parse_line(line) is None, f'should not have parsed: {line!r}'


def test_heading_is_unwrapped_past_pi(mega):
    """heading_rad must pass +pi without wrapping; the filter differences it."""
    mega.feed(protocol.ARM)
    rows = []
    for t in range(0, 4000):
        if t % 50 == 0:
            mega.feed(protocol.format_velocity(0.0, 0.0, 2.0))
        mega.pump(t)
        rows += [r for r in (protocol.parse_line(ln)
                             for ln in mega.drain().decode().split('\r\n')) if r]
    headings = [r.heading_rad for r in rows]
    assert max(headings) > math.pi, 'did not rotate far enough to test the wrap'
    assert all(b >= a - 1e-6 for a, b in zip(headings, headings[1:])), 'heading wrapped'
