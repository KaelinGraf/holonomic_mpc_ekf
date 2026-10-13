#!/usr/bin/env python3
"""
Fit each wheel's deadband and gain from a tools/drive_calibration_run.py bag.

    python3 tools/drive_wheel_fit.py bags/drivecal_20261013_043214
    python3 tools/drive_wheel_fit.py <bag> --fixes        # translation from the board poses
    python3 tools/drive_wheel_fit.py --selftest

drive_calibration_fit.py fits one deadband and gain per axis. But single-axis steps
drive all four wheels at once, so wheels that differ show up only as coupling: on
2026-10-13 every strafe also turned the robot clockwise, because two motors start
late in one direction. This fits the four wheels instead, against every step at once:

    wheel i, asked for c_i m/s by the mixing (c = J [vx, vy, wz], as drive.cpp
    computes it with the channels rotated), gets a pulse p_i = c_i x 1600 us per
    m/s (the single-constant firmware the run used: WHEEL_MAX_MPS 0.5, 800 us)
    and turns at  g_i x (|p_i| - d_i)  past its deadband d_i (one per direction),
    in the sign of p_i. The body moves as the mixing's inverse of those speeds,
    with a strafe efficiency and an effective turn lever:

        vx = (a_lf - a_rf - a_lr + a_rr) / 4
        vy = eta (a_lf + a_rf + a_lr + a_rr) / 4
        wz = (a_lf - a_rf + a_lr - a_rr) / (4 L)

Each step's speed comes from the accelerometer, integrated over the step's start
(0 -> v) and stop (v -> 0), and its turn rate from the gyro. The board poses are
not used for translation by default: with the board head-on most were ambiguous,
and they read 0.03-0.04 m/s for strafes the accelerometer put at 0.10-0.14 m/s
(forward, the two agree to 10-20 %).

It prints the firmware's per-wheel constants for config.h (DRIVE_WHEEL_CAL). The
run must use the single-constant firmware (DRIVE_WHEEL_CAL false), since the
pulses are worked out from the commands with it. With the run's plan and noise
(the self-test), gains come back within about 2 %, deadbands within about 5 us,
and the strafe efficiency only within about 0.1: the sideways steps alone pin it,
and the integrated speeds are the noisiest measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

US_PER_MPS = 800.0 / 0.50
L_RUN = 0.1655
NAMES = ('LF', 'RF', 'LR', 'RR')
J = np.array([[1.0, 1.0, L_RUN], [-1.0, 1.0, -L_RUN], [-1.0, 1.0, L_RUN], [1.0, 1.0, -L_RUN]])
FK = np.array([[1.0, -1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]]) / 4.0
SIGMA = (0.02, 0.02, 0.01)
ACC_WINDOW_S = 0.7


def _fit_module():
    """Import drive_calibration_fit.py from beside this file (tools/ is not a package)."""
    path = Path(__file__).with_name('drive_calibration_fit.py')
    spec = importlib.util.spec_from_file_location('drive_calibration_fit', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['drive_calibration_fit'] = mod
    spec.loader.exec_module(mod)
    return mod


def softplus(x, w=4.0):
    """Return max(x, 0) smoothed over a few us, so that the deadbands have a gradient."""
    return w * np.logaddexp(0.0, x / w)


def predict(theta, cmd, smooth=True):
    """Body response [vx, vy, wz] to a commanded twist, for parameters theta (see unpack)."""
    g, d_pos, d_neg, eta, lever = unpack(theta)
    p = US_PER_MPS * (J @ cmd)
    past = np.abs(p) - np.where(p >= 0.0, d_pos, d_neg)
    a = np.sign(p) * g * (softplus(past) if smooth else np.maximum(past, 0.0))
    v = FK @ a
    return np.array([v[0], eta * v[1], v[2] / lever])


def unpack(theta):
    """Split theta: [gain x4 (mm/s per us), deadband + x4, deadband - x4 (us), eta, lever (m)]."""
    return theta[0:4] * 1e-3, theta[4:8], theta[8:12], theta[12], theta[13]


def fit(rows):
    """Least squares over every step; rows are (cmd twist, measured [vx, vy, wz])."""
    from scipy.optimize import least_squares

    def resid(theta):
        out = []
        for cmd, meas in rows:
            pr = predict(theta, cmd)
            out += [(pr[k] - meas[k]) / SIGMA[k] for k in range(3) if math.isfinite(meas[k])]
        return np.array(out)

    theta0 = np.array([1.2] * 4 + [50.0] * 8 + [1.0, L_RUN])
    lo = np.array([0.2] * 4 + [0.0] * 8 + [0.2, 0.05])
    hi = np.array([5.0] * 4 + [300.0] * 8 + [3.0, 0.4])
    rng = np.random.default_rng(0)
    best = None
    for trial in range(40):
        scatter = rng.uniform(0.5, 1.8, theta0.size)
        start = theta0 if trial == 0 else np.clip(theta0 * scatter, lo, hi)
        r = least_squares(resid, start, bounds=(lo, hi), x_scale='jac')
        if best is None or r.cost < best.cost:
            best = r
    return best.x, math.sqrt(float(np.mean(best.fun ** 2)))


def imu_speeds(bag, steps):
    """Each step's [vx, vy] from the accelerometer: the change at its start, less at its stop."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topic = next(name for name in types if name.endswith('imu/data'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    cls = get_message(types[topic])
    rows = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        m = deserialize_message(data, cls)
        rows.append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec,
                     m.linear_acceleration.x, m.linear_acceleration.y))
    acc = np.array(rows)

    def integrate(a, b, bias):
        s = (acc[:, 0] >= a) & (acc[:, 0] <= b)
        if s.sum() < 3:
            return np.full(2, math.nan)
        return np.trapezoid(acc[s, 1:3] - bias, acc[s, 0], axis=0)

    out = []
    for st in steps:
        before = (acc[:, 0] >= st.t0 - 0.8) & (acc[:, 0] <= st.t0 - 0.05)
        after = (acc[:, 0] >= st.t1 + 0.9) & (acc[:, 0] <= st.t1 + 1.8)
        b0 = acc[before, 1:3].mean(axis=0)
        b1 = acc[after, 1:3].mean(axis=0) if after.sum() > 10 else b0
        out.append(0.5 * (integrate(st.t0, st.t0 + ACC_WINDOW_S, b0)
                          - integrate(st.t1, st.t1 + ACC_WINDOW_S, b1)))
    return out


def report(theta, rms, rows, labels):
    """Print the wheels, each step against the model, and the config.h lines."""
    g, d_pos, d_neg, eta, lever = unpack(theta)
    print(f'per-wheel fit: residual rms {rms:.2f} sigma over {len(rows)} steps')
    for i, n in enumerate(NAMES):
        print(f'  {n}: {g[i] * 1e3:.3f} mm/s per us, deadband {d_pos[i]:5.1f} us (+) '
              f'{d_neg[i]:5.1f} us (-)')
    print(f'  strafe efficiency {eta:.3f}, effective turn lever {lever:.4f} m')
    print('step          measured vx vy wz          model vx vy wz')
    for (cmd, meas), lab in zip(rows, labels):
        pr = predict(theta, cmd, smooth=False)
        print(f'  {lab:12s} {meas[0]:+.3f} {meas[1]:+.3f} {meas[2]:+.3f}    '
              f'{pr[0]:+.3f} {pr[1]:+.3f} {pr[2]:+.3f}')
    print('\nconfig.h, with DRIVE_WHEEL_CAL true:')
    print(f'  constexpr float CHASSIS_L_PLUS_W = {lever:.4f}f;')
    print(f'  constexpr float STRAFE_GAIN = {1.0 / eta:.2f}f;')
    print('  constexpr float WHEEL_DEADBAND_US[4][2] = {\n    '
          + ', '.join(f'{{{d_pos[i]:.1f}f, {d_neg[i]:.1f}f}}' for i in range(4)) + '};')
    print('  constexpr float WHEEL_MPS_PER_US[4] = {'
          + ', '.join(f'{g[i]:.3e}f' for i in range(4)) + '};')


def selftest():
    """Recover known wheels from simulated steps, with the run's plan and noise."""
    rng = np.random.default_rng(5)
    true = np.array([1.3, 1.5, 1.6, 1.35, 51, 56, 98, 55, 100, 61, 58, 74, 0.9, 0.112])
    rows = []
    plan = (('x', (0.02, 0.04, 0.06, 0.1, 0.15)), ('y', (0.02, 0.04, 0.06, 0.1, 0.15)),
            ('r', (0.05, 0.1, 0.2, 0.3, 0.5)))
    for axis, speeds in plan:
        for s in speeds:
            for sign in (1.0, -1.0):
                cmd = np.zeros(3)
                cmd['xyr'.index(axis)] = sign * s
                meas = predict(true, cmd, smooth=False) + rng.normal(0.0, SIGMA)
                rows.append((cmd, meas))
    theta, rms = fit(rows)
    g, d_pos, d_neg, eta, lever = unpack(theta)
    tg, tdp, tdn, teta, tlev = unpack(true)
    ok = (np.allclose(g, tg, rtol=0.15) and np.allclose(d_pos, tdp, atol=15)
          and np.allclose(d_neg, tdn, atol=15) and abs(eta - teta) < 0.15
          and abs(lever - tlev) < 0.015)
    print(f'selftest: rms {rms:.2f} sigma; gains {np.round(g * 1e3, 2)} (true {tg * 1e3}); '
          f'deadbands {np.round(d_pos)} / {np.round(d_neg)}; eta {eta:.2f}; lever {lever:.3f}')
    print('selftest', 'passed' if ok else 'FAILED')
    return ok


def main():
    """Fit a bag, or run the self-test."""
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('bag', nargs='?', help='a drivecal_* bag directory')
    ap.add_argument('--fixes', action='store_true',
                    help='translation from the board poses instead of the accelerometer')
    ap.add_argument('--settle', type=float, default=0.5, help='for the gyro and pose windows')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.bag:
        ap.error('give a bag, or --selftest')
    dcf = _fit_module()
    d = dcf.read_bag(Path(args.bag))
    steps = dcf.find_steps(d['cmd'])
    dcf.measure(steps, d, dcf.Firmware(), args.settle)
    speeds = None if args.fixes else imu_speeds(Path(args.bag), steps)
    rows, labels = [], []
    for k, st in enumerate(steps):
        cmd = np.zeros(3)
        cmd['xyr'.index(st.axis)] = st.cmd
        v = (st.fwd, st.left) if args.fixes else speeds[k]
        rows.append((cmd, np.array([v[0], v[1], st.r_imu])))
        labels.append(f'{st.axis} {st.cmd:+.3f}')
    theta, rms = fit(rows)
    report(theta, rms, rows, labels)


if __name__ == '__main__':
    main()
