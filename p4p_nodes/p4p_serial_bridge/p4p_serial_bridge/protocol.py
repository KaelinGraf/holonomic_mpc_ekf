"""Wire format of the p4p_arduino serial link.

No ROS and no serial port in here on purpose: the parsing and formatting are the
part most likely to be wrong, and this way they can be tested against
tools/fake_mega.py on a machine with no ROS installed.

The authority for all of it is p4p_arduino/README.md plus command.cpp. Uplink is
one CSV row per line at 50 Hz; downlink is E / S / V,<vx>,<vy>,<wz>.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

TELEMETRY_HEADER = ('t_ms,heading_rad,yaw_rate_dps,vx,vy,wz,flags,resets,bad,'
                    'req_vx,req_vy,req_wz,us_lf,us_rf,us_lr,us_rr')
LEGACY_TELEMETRY_HEADER = 't_ms,heading_rad,yaw_rate_dps,vx,vy,wz,flags,resets,bad'
FIELDS = 16
LEGACY_FIELDS = 9

BOOT_BANNER = '# p4p_arduino starting'

FLAG_ARMED = 0x01
FLAG_TIMEOUT = 0x02
FLAG_SATURATED = 0x04

FIRMWARE_CMD_TIMEOUT_S = 0.400

MAX_LINE_LEN = 47

ARM = b'E\n'
DISARM = b'S\n'


@dataclass(frozen=True)
class Telemetry:
    """One decoded telemetry row, with every field exactly as the Mega reported it.

    Args:
        t_ms: Mega millis() at the last IMU sample [ms].
        heading_rad: Continuous, unwrapped IMU yaw [rad].
        yaw_rate_dps: Calibrated gyro Z rate [deg/s].
        applied: Twist (vx, vy, wz) the chassis acted on, after saturation [m/s, m/s, rad/s].
        flags: Status bitfield of FLAG_ARMED, FLAG_TIMEOUT and FLAG_SATURATED.
        imu_resets: Count of spontaneous BNO085 resets.
        bad_commands: Count of malformed command lines the Mega rejected.
        requested: Last V the Mega accepted, before saturation; None from legacy firmware.
        pulse_us: Pulse widths on channels lf, rf, lr, rr [us]; None from legacy firmware.
    """

    t_ms: int
    heading_rad: float
    yaw_rate_dps: float
    applied: tuple[float, float, float]
    flags: int
    imu_resets: int
    bad_commands: int
    requested: tuple[float, float, float] | None = None
    pulse_us: tuple[int, int, int, int] | None = None

    @property
    def armed(self) -> bool:
        """Report whether the firmware has had an E and no S since."""
        return bool(self.flags & FLAG_ARMED)

    @property
    def timed_out(self) -> bool:
        """Report whether the firmware's 400 ms command watchdog has expired."""
        return bool(self.flags & FLAG_TIMEOUT)

    @property
    def saturated(self) -> bool:
        """Report whether the last command was scaled down to fit the wheels."""
        return bool(self.flags & FLAG_SATURATED)


def is_header(line: str) -> bool:
    """Report whether a line is a CSV header, of this firmware or any other.

    Args:
        line: One received line.
    """
    return line.strip().split(',', 1)[0] == 't_ms'


def parse_line(line: str) -> Telemetry | None:
    """Decode one telemetry row, or return None when the line is not one.

    Status lines, headers, blanks, wrong field counts and non-finite numbers all give None.

    Args:
        line: One received line, with or without its line ending.
    """
    s = line.strip()
    if not s or s.startswith('#') or is_header(s):
        return None
    parts = s.split(',')
    if len(parts) not in (FIELDS, LEGACY_FIELDS):
        return None
    try:
        t_ms = int(parts[0])
        heading, yaw_dps, vx, vy, wz = (float(p) for p in parts[1:6])
        flags, resets, bad = (int(p) for p in parts[6:9])
        requested = pulse_us = None
        if len(parts) == FIELDS:
            requested = tuple(float(p) for p in parts[9:12])
            pulse_us = tuple(int(p) for p in parts[12:16])
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (heading, yaw_dps, vx, vy, wz, *(requested or ()))):
        return None
    if pulse_us is not None and not all(0 <= us <= 0xFFFF for us in pulse_us):
        return None
    return Telemetry(t_ms=t_ms, heading_rad=heading, yaw_rate_dps=yaw_dps,
                     applied=(vx, vy, wz), flags=flags,
                     imu_resets=resets, bad_commands=bad,
                     requested=requested, pulse_us=pulse_us)


def format_velocity(vx: float, vy: float, wz: float) -> bytes:
    """Encode a body twist as a V line.

    Raise ValueError for a non-finite twist or a line over the firmware's length limit.

    Args:
        vx: Forward velocity [m/s].
        vy: Leftward velocity [m/s].
        wz: Yaw rate, anticlockwise positive [rad/s].
    """
    if not all(math.isfinite(v) for v in (vx, vy, wz)):
        raise ValueError(f'non-finite twist: {vx}, {vy}, {wz}')
    line = f'V,{vx:.3f},{vy:.3f},{wz:.3f}\n'
    if len(line) - 1 > MAX_LINE_LEN:
        raise ValueError(f'V line is {len(line) - 1} bytes, over the firmware limit: {line!r}')
    return line.encode('ascii')


def describe_flags(flags: int) -> str:
    """Render the flags bitfield for a log line.

    Args:
        flags: Telemetry status bitfield.
    """
    names = [n for bit, n in ((FLAG_ARMED, 'ARMED'), (FLAG_TIMEOUT, 'TIMEOUT'),
                              (FLAG_SATURATED, 'SAT')) if flags & bit]
    return '|'.join(names) if names else 'DISARMED'


class LineReader:
    """Reassembler of newline-terminated lines from arbitrary serial chunks.

    Args:
        max_len: Longest partial line kept before it is dropped as noise [bytes].
    """

    def __init__(self, max_len: int = 1024) -> None:
        self._buf = bytearray()
        self._max_len = max_len
        self.overruns = 0

    def feed(self, data: bytes) -> list[str]:
        """Add received bytes and return whatever complete lines that produced.

        Args:
            data: Bytes read from the serial port.
        """
        self._buf += data
        out: list[str] = []
        while True:
            i = self._buf.find(b'\n')
            if i < 0:
                break
            raw = bytes(self._buf[:i])
            del self._buf[:i + 1]
            out.append(raw.decode('ascii', 'replace').rstrip('\r'))
        if len(self._buf) > self._max_len:
            self.overruns += 1
            self._buf.clear()
        return out

    def reset(self) -> None:
        """Discard any partial line, after a reconnect or a firmware reset."""
        self._buf.clear()
