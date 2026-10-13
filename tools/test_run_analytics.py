"""
Test run_analytics on synthetic data: the fits, the timeline, the per-fix NIS and the old telemetry layout.

    python3 -m pytest tools/test_run_analytics.py -q
"""
from pathlib import Path
import math
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analytics as ra  # noqa: E402, I100


def test_lag_and_gain_are_recovered():
    """A drive that delivers 0.6x the command 150 ms late, with noise: the fit finds both."""
    rng = np.random.default_rng(0)
    tc = np.arange(0.0, 20.0, 0.05)
    cmd = 0.1 * np.sign(np.sin(2 * np.pi * tc / 4.0))
    t = np.arange(1.0, 19.0, 0.05)
    meas = 0.6 * np.interp(t - 0.15, tc, cmd) + rng.normal(0, 0.005, len(t))
    r = ra.fit_lag_gain(t, tc, cmd, meas)
    assert r['gain'] == pytest.approx(0.6, abs=0.02)
    assert r['lag_s'] == pytest.approx(0.15, abs=0.011)
    assert r['r2'] > 0.9 and 'note' not in r and 'flag' not in r


def test_a_drive_turning_the_wrong_way_is_flagged_not_dropped():
    tc = np.arange(0.0, 10.0, 0.05)
    cmd = 0.3 * np.sin(2 * np.pi * tc / 3.0)
    t = np.arange(0.5, 9.5, 0.05)
    r = ra.fit_lag_gain(t, tc, cmd, -0.25 * np.interp(t - 0.1, tc, cmd), min_range=0.05)
    assert r['gain'] == pytest.approx(-0.25, abs=0.01)
    assert r['flag'] == 'moved against the command' and 'note' not in r


def test_too_little_motion_is_not_fitted():
    t = np.arange(0.0, 10.0, 0.05)
    r = ra.fit_lag_gain(t, t, np.full(len(t), 0.005), np.zeros(len(t)) + 0.001 * np.sin(t))
    assert r.get('note')


def test_window_fit_gives_slope_and_centre_value():
    t = np.arange(0.0, 5.0, 1 / 15)
    y = 0.2 * t - 0.5
    slope, value = ra.window_fit(t, y, np.array([1.0, 2.5]), 0.2)
    assert slope == pytest.approx([0.2, 0.2]) and value == pytest.approx([-0.3, 0.0])
    slope, _ = ra.window_fit(t, y, np.array([10.0]), 0.2)
    assert np.isnan(slope[0])


def test_held_command_drops_to_zero_after_the_bridge_timeout():
    tc = np.array([0.0, 0.05, 0.10])
    v = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    out = ra.held(tc, v, np.array([-0.01, 0.07, 0.15, 0.25]))
    assert out[:, 0] == pytest.approx([0.0, 0.1, 0.2, 0.0])


def stage_rows(spec):
    """(stage, seconds) pairs to dock/stage rows at 20 Hz."""
    rows, t = [], 100.0
    for s, dur in spec:
        for _ in range(int(round(dur * 20))):
            rows.append((t, s, 0, 0, 0, 0.18, 0.3, float(s in ra.MOVING)))
            t += 0.05
    return np.array(rows, float)


def test_timeline_of_a_docked_run():
    b = ra.Bag(name='t', path=Path('.'))
    b.stage = stage_rows([(ra.IDLE, 0.5), (ra.ACQUIRE, 2.0), (ra.APPROACH, 3.0), (ra.FINAL, 4.0), (ra.DOCKED, 5.0),
                          (ra.IDLE, 0.5)])
    b.start, b.end = 100.0, 115.0
    tl = ra.timeline(b)
    assert tl.outcome == 'DOCKED'
    assert tl.approach - tl.t0 == pytest.approx(2.0, abs=0.06)
    assert tl.docked - tl.approach == pytest.approx(7.0, abs=0.06)
    assert tl.t_cmd == pytest.approx(tl.docked)


def test_a_run_disarmed_in_final():
    b = ra.Bag(name='t', path=Path('.'))
    b.stage = stage_rows([(ra.ACQUIRE, 1.0), (ra.APPROACH, 2.0), (ra.FINAL, 3.0), (ra.IDLE, 1.0)])
    b.start, b.end = 100.0, 107.0
    tl = ra.timeline(b)
    assert tl.outcome == 'disarmed in FINAL' and not math.isfinite(tl.docked)


def test_each_change_of_nis_or_verdict_is_one_processed_fix():
    b = ra.Bag(name='t', path=Path('.'))
    nis = [np.nan] * 3 + [2.0] * 4 + [150.0] * 3 + [150.0] * 2 + [1.5] * 3
    acc = [0] * 3 + [1] * 4 + [0] * 3 + [1] * 2 + [1] * 3
    e = np.zeros((len(nis), 17))
    e[:, 0] = e[:, 1] = 100.0 + 0.01 * np.arange(len(nis))
    e[:, 15], e[:, 16] = nis, acc
    b.est = e
    b.stage = stage_rows([(ra.APPROACH, 1.0)])
    f = np.zeros((4, 20))
    f[:, 0] = f[:, 1] = [100.025, 100.065, 100.095, 100.115]
    f[:, 19] = [0, 1, 1, 0]
    b.fix = f
    pf = ra.processed_fixes(b)
    assert pf[:, 1] == pytest.approx([2.0, 150.0, 150.0, 1.5]) and list(pf[:, 2]) == [1, 0, 1, 1]
    assert list(pf[:, 4]) == [0, 1, 1, 0]


def test_old_telemetry_layout_is_decoded():
    """A DriveTelemetry from before 876cd4a: CDR, header, mega_t_ms, two doubles, a twist, flags, two uint16."""
    frame = b'base_link\x00'
    body = struct.pack('<iI', 1791863390, 500_000_000) + struct.pack('<I', len(frame)) + frame
    body += b'\x00' * (-len(body) % 4) + struct.pack('<I', 12345)
    body += b'\x00' * (-len(body) % 8) + struct.pack('<dd', 1.5, -0.25)
    body += struct.pack('<6d', 0.05, -0.02, 0.0, 0.0, 0.0, 0.3) + struct.pack('<4B', 1, 0, 1, 5)
    body += b'\x00' * (-len(body) % 2) + struct.pack('<HH', 0, 2)
    row = ra.old_telemetry(b'\x00\x01\x00\x00' + body)
    assert row[0] == pytest.approx(1791863390.5)
    assert row[1:4] == pytest.approx((0.05, -0.02, 0.3)) and math.isnan(row[4])
    assert row[7:11] == (1.0, 0.0, 1.0, 5) and row[-1] == pytest.approx(-0.25)
    assert ra.old_telemetry(b'\x00\x01\x00\x00' + body + b'\x00' * 8) is None


def test_overshoot_is_measured_past_the_centre_line_away_from_the_start():
    t = np.arange(0.0, 6.0, 0.05)
    y = np.where(t < 3.0, -0.2 + 0.09 * t, 0.07 - 0.02 * (t - 3.0))
    m = ra.motion_metrics(t, np.linspace(-0.6, 0.0, len(t)), y, np.zeros(len(t)), 0.3, 6.0)
    assert m['start']['y_m'] == pytest.approx(-0.2 + 0.09 * 0.3 / 2, abs=0.02)
    assert m['lateral_overshoot_mm'] == pytest.approx(70.0, abs=1.0)


def test_mismatches_compare_only_what_the_bag_shows():
    p = {'sup': {'standoff': 0.30}, 'lim': {'v_approach': 0.18, 'r_max': 0.3, 'v_final': 0.03},
         'mpc': {'Np': 80}, 'geom': {'marker_m': [0.326, 0.0], 'cam_xyz': [0.07, 0.0, 0.08]}}
    assert ra.mismatches({'standoff_m': 0.3, 'Np': 80}, p) == []
    assert ra.mismatches({'v_final': 0.09, 'marker_x_m': 0.236}, p) == ['v_final 0.09 (config 0.03)',
                                                                        'marker_x_m 0.236 (config 0.326)']
