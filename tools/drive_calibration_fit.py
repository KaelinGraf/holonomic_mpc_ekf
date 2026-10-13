#!/usr/bin/env python3
"""
Fit each drive axis's deadband and gain from a tools/drive_calibration_run.py bag.

    python3 tools/drive_calibration_fit.py bags/drivecal_20261013_181500
    python3 tools/drive_calibration_fit.py <bag> --settle 0.6 --csv steps.csv
    python3 tools/drive_calibration_fit.py --selftest

Each single-axis step on cmd_vel is measured over its steady part, from --settle s after the
command to its end:

  forward, left   a straight line through the fixes' x and y in the target frame (fiducial/fix,
                  capture-stamped; ambiguous ones only when too few others), turned into the body
                  frame by the fixes' heading
  turn rate       imu/data's gyro z, less its bias over the still spells between the steps; the
                  fixes' heading gives a second value, for comparison
  pulses          drive/telemetry's pulse_us (firmware from 2026-10-13 on), else worked out from
                  the command with config.h's constants (--wheel-max-mps, --l-plus-w, ...)

Then, per axis, the response along the command against the command:

  deadband        the largest command with no measurable motion (4 x the noise of the same
                  measurement over the still spells, or a floor), and the smallest that moved
  offset + gain   a straight line through the steps that moved
  dead zone       response = gain * (|command| - deadband) through every step, each way too

and the same against the pulse width's distance from neutral, which is where the deadband lives:
a motor does nothing within some microseconds of 1500 whatever the axis asked for. The forward
steps give that pulse deadband and the wheels' speed per microsecond (they roll without slip);
the strafe and turn steps give their own speed per microsecond against it, hence STRAFE_GAIN and
the turn's effective radius. From those it suggests config.h values two ways: with a pulse
deadband offset added to the firmware (one change that is right for any mix of axes, since each
wheel gets its own), or constants alone, with a minimum command per axis for the controller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np

AXES = 'xyr'
NAMES = {'x': 'forward', 'y': 'left', 'r': 'turn'}
UNITS = {'x': 'm/s', 'y': 'm/s', 'r': 'rad/s'}
FLOORS = {'x': 0.004, 'y': 0.004, 'r': 0.015}
NEUTRAL_US = 1500
FLAG_ARMED, FLAG_TIMEOUT, FLAG_SATURATED = 0x01, 0x02, 0x04


@dataclass(frozen=True)
class Firmware:
    """config.h as flashed, for working out pulses the telemetry did not report."""

    wheel_max_mps: float = 0.50
    l_plus_w: float = 0.1655
    strafe_gain: float = 1.0
    span_us: float = 800.0

    def pulse_offset(self, axis: str, cmd: float) -> float:
        """Distance of every wheel's pulse from neutral for a single-axis command [us]."""
        per = {'x': 1.0, 'y': self.strafe_gain, 'r': self.l_plus_w}[axis]
        return min(abs(cmd) * per / self.wheel_max_mps, 1.0) * self.span_us


@dataclass
class Step:
    axis: str
    cmd: float
    t0: float
    t1: float
    n_fix: int = 0
    ambiguous: bool = False
    fwd: float = math.nan
    left: float = math.nan
    r_imu: float = math.nan
    r_fix: float = math.nan
    pulses: tuple = ()
    u: float = math.nan
    flags: int = 0
    driving: bool = True
    notes: list = field(default_factory=list)

    @property
    def response(self) -> float:
        """Return the motion along the commanded axis, positive in the commanded direction."""
        v = {'x': self.fwd, 'y': self.left,
             'r': self.r_imu if math.isfinite(self.r_imu) else self.r_fix}[self.axis]
        return v * math.copysign(1.0, self.cmd)

    @property
    def usable(self) -> bool:
        return math.isfinite(self.response) and self.driving


def stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def read_bag(path: Path) -> dict:
    """Read the bag's cmd_vel, fixes, gyro and telemetry into arrays (see measure())."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    keys = {'cmd_vel': 'cmd', 'fiducial/fix': 'fix', 'imu/data': 'imu', 'drive/telemetry': 'tel'}
    which = {name: k for name in types for suffix, k in keys.items() if name.endswith(suffix)}
    cls = {name: get_message(types[name]) for name in which}
    out = {k: [] for k in keys.values()}
    unreadable = 0
    while reader.has_next():
        topic, data, t_recv = reader.read_next()
        if topic not in cls:
            continue
        try:
            m = deserialize_message(data, cls[topic])
        except Exception:
            unreadable += 1
            continue
        k = which[topic]
        if k == 'cmd':
            out[k].append((t_recv * 1e-9, m.linear.x, m.linear.y, m.angular.z))
        elif k == 'fix':
            out[k].append((stamp(m), m.x, m.y, m.theta, float(m.ambiguous), m.range))
        elif k == 'imu':
            out[k].append((stamp(m), m.angular_velocity.z))
        else:
            pulses = tuple(int(p) for p in getattr(m, 'pulse_us', (0, 0, 0, 0)))
            out[k].append((stamp(m), *pulses, m.flags))
    if unreadable:
        print(f'{unreadable} messages did not deserialise with the installed definitions')
    tel_cls = [cls[t] for t in cls if which[t] == 'tel']
    if out['tel'] and not any('pulse_us' in c.get_fields_and_field_types() for c in tel_cls):
        print('this convchart_interfaces has no pulse_us, so the logged pulses cannot be read '
              '(rebuild it); working them out from the command instead')
    shapes = {'cmd': 4, 'fix': 6, 'imu': 2, 'tel': 6}
    return {k: np.array(sorted(v), float).reshape(-1, shapes[k]) for k, v in out.items()}


def find_steps(cmd: np.ndarray, min_s: float = 0.5, gap_s: float = 0.25) -> list[Step]:
    """Return the runs of one command on one axis, min_s or longer, in cmd_vel (t, vx, vy, wz)."""
    steps, i, n = [], 0, len(cmd)
    while i < n:
        j = i + 1
        while (j < n and np.array_equal(cmd[j, 1:], cmd[i, 1:])
               and cmd[j, 0] - cmd[j - 1, 0] <= gap_s):
            j += 1
        nz = np.flatnonzero(np.abs(cmd[i, 1:]) > 1e-9)
        t1 = cmd[j, 0] if j < n else cmd[j - 1, 0] + 0.02
        if len(nz) == 1 and t1 - cmd[i, 0] >= min_s:
            axis = int(nz[0])
            steps.append(Step(AXES[axis], float(cmd[i, 1 + axis]), float(cmd[i, 0]), float(t1)))
        i = j
    return steps


def still_spells(cmd: np.ndarray, after_s: float = 1.0, min_s: float = 0.8) -> list:
    """Return (t0, t1) of the zero-command spells, from after_s in (once the robot stopped)."""
    spells, i, n = [], 0, len(cmd)
    while i < n:
        j = i + 1
        while j < n and np.array_equal(cmd[j, 1:], cmd[i, 1:]):
            j += 1
        if not np.any(cmd[i, 1:]):
            t0, t1 = cmd[i, 0] + after_s, cmd[j - 1, 0]
            if t1 - t0 >= min_s:
                spells.append((float(t0), float(t1)))
        i = j
    return spells


def robust_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Slope of a straight line through (t, y), refitted once without points 3 MADs off it."""
    if len(t) < 3 or np.ptp(t) <= 0:
        return math.nan
    tc = t - t.mean()
    p = np.polyfit(tc, y, 1)
    r = y - np.polyval(p, tc)
    keep = np.abs(r) <= max(3.0 * 1.4826 * float(np.median(np.abs(r))), 1e-3)
    if 3 <= keep.sum() < len(t):
        p = np.polyfit(tc[keep], y[keep], 1)
    return float(p[0])


def fix_motion(fix: np.ndarray, t0: float, t1: float, min_clear: int = 5):
    """Return (n, used ambiguous, forward, left, heading rate) from the fixes in [t0, t1]."""
    sel = fix[(fix[:, 0] >= t0) & (fix[:, 0] <= t1)]
    clear = sel[sel[:, 4] == 0]
    used_amb = len(clear) < min_clear
    sel = sel if used_amb else clear
    if len(sel) < 3:
        return len(sel), used_amb, math.nan, math.nan, math.nan
    t = sel[:, 0]
    th = np.unwrap(sel[:, 3])
    vx, vy = robust_slope(t, sel[:, 1]), robust_slope(t, sel[:, 2])
    c, s = math.cos(float(np.median(th))), math.sin(float(np.median(th)))
    return len(sel), used_amb, c * vx + s * vy, -s * vx + c * vy, robust_slope(t, th)


def measure(steps: list[Step], d: dict, fw: Firmware, settle: float) -> dict:
    """Fill each step's measurements in place; return the still spells' statistics."""
    imu, fix, tel = d['imu'], d['fix'], d['tel']
    spells = still_spells(d['cmd'])
    still = [imu[(imu[:, 0] >= a) & (imu[:, 0] <= b), 1] for a, b in spells]
    gyro_still = np.concatenate(still) if still else np.zeros(0)
    bias = float(np.median(gyro_still)) if len(gyro_still) else 0.0
    have_pulses = len(tel) and np.any(tel[:, 1:5])
    for s in steps:
        a, b = s.t0 + settle, s.t1
        s.n_fix, s.ambiguous, s.fwd, s.left, s.r_fix = fix_motion(fix, a, b)
        g = imu[(imu[:, 0] >= a) & (imu[:, 0] <= b), 1]
        s.r_imu = float(np.mean(g) - bias) if len(g) else math.nan
        s.u = fw.pulse_offset(s.axis, s.cmd)
        rows = tel[(tel[:, 0] >= a) & (tel[:, 0] <= b)] if len(tel) else tel
        if len(rows):
            s.flags = int(np.bitwise_or.reduce(rows[:, 5].astype(int)))
            if have_pulses and np.all(rows[:, 1:5]):
                s.pulses = tuple(int(v) for v in np.median(rows[:, 1:5], axis=0))
                u = float(np.mean(np.abs(np.array(s.pulses) - NEUTRAL_US)))
                if abs(u - s.u) > 2.0:
                    s.notes.append(f'pulses {u:.0f} us off neutral, constants say {s.u:.0f}')
                s.u = u
        if len(rows) and (s.flags & FLAG_TIMEOUT or not s.flags & FLAG_ARMED):
            s.driving = False
            s.notes.append('the Mega was not driving (disarmed or timed out): left out')
        if s.flags & FLAG_SATURATED:
            s.notes.append('saturated')
        if s.ambiguous and s.axis != 'r':
            s.notes.append('ambiguous fixes')
    span = max(float(np.median([s.t1 - s.t0 for s in steps])) - settle, 0.5) if steps else 1.0
    est = {'x': [], 'y': [], 'r': []}
    for a, b in spells:
        n = max(1, int((b - a) // span))
        for w0, w1 in zip(np.linspace(a, b, n + 1)[:-1], np.linspace(a, b, n + 1)[1:]):
            _, _, f, left, _ = fix_motion(fix, w0, w1)
            g = imu[(imu[:, 0] >= w0) & (imu[:, 0] <= w1), 1]
            est['x'].append(f)
            est['y'].append(left)
            est['r'].append(float(np.mean(g)) - bias if len(g) else math.nan)
    noise = {}
    for k, v in est.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        noise[k] = float(np.sqrt(np.mean(v * v))) if len(v) >= 3 else math.nan
    thr = {k: max(FLOORS[k], 4.0 * noise[k]) if math.isfinite(noise[k]) else FLOORS[k]
           for k in AXES}
    return {'bias': bias, 'spells': len(spells), 'noise': noise, 'thr': thr,
            'pulses_logged': bool(have_pulses)}


def deadzone_fit(c: np.ndarray, s: np.ndarray):
    """Fit s = g * max(c - d, 0) for c >= 0, 0 <= d < max(c). Returns (d, g, rms), or None."""
    if len(c) < 2 or c.max() <= 0:
        return None
    best = None
    for d in np.linspace(0.0, c.max(), 801)[:-1]:
        h = np.maximum(c - d, 0.0)
        g = float(h @ s) / float(h @ h)
        rms = float(np.sqrt(np.mean((s - g * h) ** 2)))
        if best is None or rms < best[2] - 1e-15:
            best = (float(d), g, rms)
    return best


def line_fit(c: np.ndarray, s: np.ndarray):
    """Fit s = offset + gain * c; (offset, gain), or None with fewer than two distinct c."""
    if len(np.unique(c)) < 2:
        return None
    gain, offset = np.polyfit(c, s, 1)
    return float(offset), float(gain)


def fit_axis(steps: list[Step], thr: float) -> dict:
    """Fit one axis's usable steps in command space and in pulse space."""
    st = [s for s in steps if s.usable]
    if not st:
        return {}
    c = np.array([abs(s.cmd) for s in st])
    resp = np.array([s.response for s in st])
    u = np.array([s.u for s in st])
    moved = resp >= thr
    out = {'n': len(st), 'thr': thr,
           'still_max': float(c[~moved].max()) if (~moved).any() else 0.0,
           'moved_min': float(c[moved].min()) if moved.any() else math.nan,
           'against': [s for s in st if s.response <= -thr],
           'deadzone': deadzone_fit(c, resp), 'line': line_fit(c[moved], resp[moved]),
           'deadzone_u': deadzone_fit(u, resp)}
    for sign, name in ((1, 'pos'), (-1, 'neg')):
        k = np.array([math.copysign(1, s.cmd) == sign for s in st])
        out[name] = deadzone_fit(c[k], resp[k]) if k.sum() >= 2 else None
    return out


def suggest(fits: dict, fw: Firmware) -> list[str]:
    """Suggest config.h constants from the pulse-space fits."""
    dz = {a: fits.get(a, {}).get('deadzone_u') for a in AXES}
    if dz['x'] is not None:
        u0, k_x, src = dz['x'][0], dz['x'][1], 'the forward steps'
    elif dz['r'] is not None:
        u0, k_x, src = dz['r'][0], dz['r'][1] * fw.l_plus_w, 'the turns, at the geometric radius'
    else:
        return ['too little forward or turn data to suggest constants']
    if k_x <= 0:
        return [f'{src} moved against the command or not at all: check the drive first']
    span = fw.span_us
    sg = k_x / dz['y'][1] if dz['y'] and dz['y'][1] > 0 else math.nan
    l_eff = k_x / dz['r'][1] if dz['x'] and dz['r'] and dz['r'][1] > 0 else math.nan
    others = [f'{NAMES[a]} {dz[a][0]:.0f}' for a in 'yr'
              if dz[a] and (a != 'r' or dz['x'])]
    out = [f'Pulse deadband {u0:.0f} us each side of neutral, from {src}'
           + (f' ({", ".join(others)} us from the others)' if others else '')
           + f'; above it a wheel gains {k_x * 1e3:.3f} mm/s per us.']
    both = []
    if math.isfinite(sg):
        both.append(f'STRAFE_GAIN       = {sg:.3f}   (was {fw.strafe_gain:g}; sideways the '
                    f'chassis makes {1 / sg:.0%} of its forward speed per us)')
    l_new = fw.l_plus_w
    if math.isfinite(l_eff):
        l_new = l_eff
        both.append(f'CHASSIS_L_PLUS_W  = {l_eff:.4f}  (was {fw.l_plus_w:.4f}, the geometry, '
                    f'with which turns run at {fw.l_plus_w / l_eff:.2f} x the command)')
    w_a, w_b = k_x * (span - u0), k_x * span
    out += ['A. Offset the pulses in the firmware, which is right for any mix of axes since each '
            'wheel gets its own:',
            f'     drive.cpp toPulse(), for norm != 0: us = NEUTRAL +/- (MOTOR_US_DEADBAND + '
            f'|norm| * ({span:.0f} - MOTOR_US_DEADBAND))',
            f'     MOTOR_US_DEADBAND = {u0:.0f}       (new)',
            f'     WHEEL_MAX_MPS     = {w_a:.3f}   (was {fw.wheel_max_mps:.3f})',
            *('     ' + b for b in both),
            'B. Constants alone, the deadband left in:',
            f'     WHEEL_MAX_MPS     = {w_b:.3f}   (was {fw.wheel_max_mps:.3f})',
            *('     ' + b for b in both)]
    mins = [f'|vx| {u0 * w_b / span:.3f} m/s']
    if math.isfinite(sg):
        mins.append(f'|vy| {dz["y"][0] * w_b / (span * sg):.3f} m/s')
    if dz['r'] is not None and dz['x'] is not None:
        mins.append(f'|wz| {dz["r"][0] * w_b / (span * l_new):.3f} rad/s')
    out += [f'   and then, one axis at a time, nothing moves below {", ".join(mins)}: the '
            'controller needs those as minimum commands.',
            'Either way, set the bridge max_linear to WHEEL_MAX_MPS and max_angular to '
            'WHEEL_MAX_MPS / CHASSIS_L_PLUS_W, and run this again to check.']
    return out


def analyse(d: dict, fw: Firmware, settle: float) -> tuple[list[Step], dict, dict]:
    """Find, measure and fit the steps: (steps, still-spell statistics, fits per axis)."""
    steps = find_steps(d['cmd'])
    info = measure(steps, d, fw, settle) if steps else {}
    fits = {a: fit_axis([s for s in steps if s.axis == a], info['thr'][a])
            for a in AXES if any(s.axis == a for s in steps)} if steps else {}
    return steps, info, fits


def report(d: dict, fw: Firmware, settle: float, csv: Path | None = None) -> None:
    steps, info, fits = analyse(d, fw, settle)
    span = d['cmd'][-1, 0] - d['cmd'][0, 0] if len(d['cmd']) else 0.0
    imu_hz = (len(d['imu']) - 1) / np.ptp(d['imu'][:, 0]) if len(d['imu']) > 1 else 0.0
    clear = int((d['fix'][:, 4] == 0).sum()) if len(d['fix']) else 0
    print(f'cmd_vel over {span:.0f} s; {len(d["fix"])} fixes ({clear} unambiguous); imu/data '
          f'{imu_hz:.0f} Hz; drive/telemetry {len(d["tel"])} rows')
    if not steps:
        sys.exit('no single-axis steps on cmd_vel: is this a drive_calibration_run.py bag?')
    print(f'{len(steps)} steps; {info["spells"]} still spells: gyro bias {info["bias"]:+.4f} '
          'rad/s, noise ' + ', '.join(f'{a} {info["noise"][a]:.4f}' for a in AXES)
          + f'; pulses {"logged" if info["pulses_logged"] else "worked out from the command"}')
    print(f'\nsteady part: {settle:g} s after each command to its end. Response = motion along '
          'the command (fixes for forward and left, the gyro for turns).')
    print(f'{"axis":>4} {"cmd":>7} {"us":>4} {"lf rf lr rr":>19} {"fwd":>7} {"left":>7} '
          f'{"r_imu":>7} {"r_fix":>7} {"resp":>7} {"n":>3}  notes')
    rows = []
    for s in steps:
        pulses = ' '.join(str(p) for p in s.pulses) if s.pulses else '-'
        print(f'{s.axis:>4} {s.cmd:+7.3f} {s.u:4.0f} {pulses:>19} {s.fwd:+7.3f} {s.left:+7.3f} '
              f'{s.r_imu:+7.3f} {s.r_fix:+7.3f} {s.response:+7.3f} {s.n_fix:3d}  '
              + '; '.join(s.notes))
        rows.append([s.axis, s.cmd, s.t0, s.t1, s.u, *(s.pulses or ('',) * 4), s.fwd, s.left,
                     s.r_imu, s.r_fix, s.response, s.n_fix, int(s.ambiguous), s.flags])
    if csv is not None:
        head = ('axis,cmd,t0,t1,u_us,us_lf,us_rf,us_lr,us_rr,fwd,left,r_imu,r_fix,response,'
                'n_fix,ambiguous,flags')
        csv.write_text(head + '\n' + '\n'.join(','.join(str(v) for v in r) for r in rows) + '\n')
        print(f'(written to {csv})')

    print('\nper axis, command space (response along the command):')
    for a, f in fits.items():
        if not f:
            print(f'  {NAMES[a]}: no usable steps')
            continue
        u = UNITS[a]
        print(f'  {NAMES[a]} ({f["n"]} steps, motion counted above {f["thr"]:.3f} {u}):')
        if not math.isfinite(f['moved_min']):
            print(f'    deadband: no step moved, up to {f["still_max"]:.3f} {u}')
        elif not f['still_max']:
            print(f'    deadband: every step moved, from {f["moved_min"]:.3f} {u}')
        else:
            print(f'    deadband: still up to {f["still_max"]:.3f} {u}, moving from '
                  f'{f["moved_min"]:.3f} {u}')
        if f['deadzone']:
            dz = f['deadzone']
            each = ', '.join(f'{w} {f[k][0]:.3f} {u} x {f[k][1]:.2f}' for k, w in
                             (('pos', '+'), ('neg', '-')) if f[k])
            print(f'    dead zone: response = {dz[1]:.2f} x (|cmd| - {dz[0]:.3f} {u}), rms '
                  f'{dz[2]:.4f}' + (f'; each way: {each}' if each else ''))
        if f['line']:
            off, gain = f['line']
            print(f'    offset + gain (moving steps): response = {off:+.4f} + {gain:.2f} x |cmd|'
                  + (f', crossing zero at {-off / gain:.3f} {u}' if gain > 0 else ''))
        if f['deadzone_u']:
            du = f['deadzone_u']
            per = 'mm/s' if a != 'r' else 'mrad/s'
            print(f'    pulse space: response = {du[1] * 1000:.3f} {per} per us x (us off neutral '
                  f'- {du[0]:.0f})')
        for s in f['against']:
            print(f'    !! {s.cmd:+.3f} {u} moved the other way ({s.response:+.3f}): check the '
                  f'sign trims')
    print('\nsuggested config.h (check against the table first):')
    for line in suggest(fits, fw):
        print('  ' + line)


def simulate(seed: int = 3, u0: float = 45.0, k_us: float = 0.0011, eta_y: float = 0.8,
             l_eff: float = 0.19, fw: Firmware = Firmware()) -> dict:
    """Simulate a drive_calibration_run.py run on a chassis of known deadband and gains."""
    rng = np.random.default_rng(seed)
    speeds = {'x': (0.02, 0.04, 0.06, 0.10, 0.15), 'y': (0.02, 0.04, 0.06, 0.10, 0.15),
              'r': (0.05, 0.1, 0.2, 0.3, 0.5)}
    sched, t = [], 3.0
    for a in AXES:
        for v in speeds[a]:
            for c in (v, -v):
                sched.append((t, t + 1.5, a, c))
                t += 3.5
    t_end, dt = t + 1.0, 0.002
    ts = np.arange(0.0, t_end, dt)
    cmd = np.zeros((len(ts), 3))
    for t0, t1, a, c in sched:
        cmd[(ts >= t0) & (ts < t1), AXES.index(a)] = c
    k_ax = {'x': k_us, 'y': eta_y * k_us, 'r': k_us / l_eff}
    target = np.zeros_like(cmd)
    for i, a in enumerate(AXES):
        u = np.array([fw.pulse_offset(a, c) for c in cmd[:, i]])
        target[:, i] = np.sign(cmd[:, i]) * k_ax[a] * np.maximum(u - u0, 0.0)
    target = np.roll(target, int(0.03 / dt), axis=0)
    vel = np.zeros_like(target)
    for j in range(1, len(ts)):
        vel[j] = vel[j - 1] + (target[j] - vel[j - 1]) * dt / 0.15
    th = 0.25 + np.cumsum(vel[:, 2]) * dt
    c, s = np.cos(th), np.sin(th)
    x = -1.0 + np.cumsum(c * vel[:, 0] - s * vel[:, 1]) * dt
    y = 0.1 + np.cumsum(s * vel[:, 0] + c * vel[:, 1]) * dt
    tc = np.arange(sched[0][0] - 1.0, t_end - 0.1, 0.02)
    ic = np.searchsorted(ts, tc)
    d = {'cmd': np.column_stack([tc, cmd[ic]])}
    tf = np.arange(0.5, t_end - 0.1, 1 / 15)
    jf = np.searchsorted(ts, tf)
    amb = rng.random(len(tf)) < 0.1
    d['fix'] = np.column_stack([tf, x[jf] + rng.normal(0, 0.0015, len(tf)),
                                y[jf] + rng.normal(0, 0.0015, len(tf)),
                                th[jf] + rng.normal(0, 0.004, len(tf)) + 0.3 * amb, amb,
                                np.hypot(x[jf], y[jf])])
    tg = np.arange(0.5, t_end - 0.1, 1 / 150)
    jg = np.searchsorted(ts, tg)
    d['imu'] = np.column_stack([tg, vel[jg, 2] + 0.004 + rng.normal(0, 0.01, len(tg))])
    tt = np.arange(0.5, t_end - 0.1, 0.02)
    jt = np.searchsorted(ts, tt)
    pulses = []
    for j in jt:
        a = next((i for i in range(3) if cmd[j, i]), None)
        off = round(fw.pulse_offset(AXES[a], cmd[j, a])) if a is not None else 0
        sg = np.sign(cmd[j, a]) if a is not None else 0
        pulses.append((NEUTRAL_US - sg * off, NEUTRAL_US - sg * off, NEUTRAL_US + sg * off,
                       NEUTRAL_US + sg * off, FLAG_ARMED))
    d['tel'] = np.column_stack([tt, np.array(pulses, float)])
    return d


def selftest() -> None:
    """Recover a known pulse deadband, gains, strafe slip and turn radius from a simulated run."""
    fw = Firmware()
    u0, k_us, eta_y, l_eff = 45.0, 0.0011, 0.8, 0.19
    steps, info, fits = analyse(simulate(u0=u0, k_us=k_us, eta_y=eta_y, l_eff=l_eff), fw, 0.5)
    assert len(steps) == 30, len(steps)
    fx, fy, fr = (fits[a]['deadzone_u'] for a in AXES)
    print(f'pulse deadband: forward {fx[0]:.1f} us, left {fy[0]:.1f}, turn {fr[0]:.1f} '
          f'(true {u0:.0f})')
    print(f'forward gain {fx[1] * 1e3:.3f} mm/s per us (true {k_us * 1e3:.3f}); left / forward '
          f'{fy[1] / fx[1]:.3f} (true {eta_y}); turn radius {fx[1] / fr[1]:.4f} m (true {l_eff})')
    for f in (fx, fy, fr):
        assert abs(f[0] - u0) < 4.0
    assert abs(fx[1] / k_us - 1) < 0.04
    assert abs(fy[1] / fx[1] - eta_y) < 0.04
    assert abs(fx[1] / fr[1] - l_eff) < 0.008
    dz = fits['x']['deadzone']
    print(f'forward dead zone in command space: {dz[0]:.4f} m/s (true '
          f'{u0 / fw.span_us * fw.wheel_max_mps:.4f}), gain {dz[1]:.3f} (true '
          f'{k_us * fw.span_us / fw.wheel_max_mps:.3f})')
    assert abs(dz[0] - u0 / fw.span_us * fw.wheel_max_mps) < 0.003
    assert abs(dz[1] / (k_us * fw.span_us / fw.wheel_max_mps) - 1) < 0.04
    assert fits['x']['still_max'] == 0.02 and fits['x']['moved_min'] == 0.04, fits['x']
    lines = suggest(fits, fw)
    assert any('MOTOR_US_DEADBAND = 4' in ln for ln in lines), lines
    print('\n'.join('  ' + ln for ln in lines))
    print('selftest passed')


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('bag', nargs='?', type=Path)
    ap.add_argument('--settle', type=float, default=0.5,
                    help='seconds after each command left out while the motors spin up')
    ap.add_argument('--csv', type=Path, help='also write the per-step table here')
    ap.add_argument('--wheel-max-mps', type=float, default=Firmware.wheel_max_mps,
                    help='config.h WHEEL_MAX_MPS when the bag was made')
    ap.add_argument('--l-plus-w', type=float, default=Firmware.l_plus_w,
                    help='config.h CHASSIS_L_PLUS_W')
    ap.add_argument('--strafe-gain', type=float, default=Firmware.strafe_gain,
                    help='config.h STRAFE_GAIN')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.bag is None:
        ap.error('give a bag, or --selftest')
    fw = Firmware(a.wheel_max_mps, a.l_plus_w, a.strafe_gain)
    plan = a.bag / 'steps.json'
    if plan.exists():
        p = json.loads(plan.read_text())
        print(f'{a.bag.name}: {len(p["steps"])} of {p["planned"]} planned steps'
              + (f', stopped early: {p["stopped"]}' if p.get('stopped') else ''))
    report(read_bag(a.bag), fw, a.settle, a.csv)


if __name__ == '__main__':
    main()
