"""The EKF against report Table 2 and the unit tests of brief Section 11."""

import math

from dock_control.ekf import Ekf, RingBuffer
from dock_control.geometry import wrap_pi
import numpy as np
import pytest

X = np.array([0.1, -0.2, 0.3, 0.05, -0.02])     # a state away from every special case
U = np.array([0.3, -0.1, 0.2])                  # [a_x, a_y, r]


def _central_differences(fun, v, h=1e-6):
    cols = []
    for i in range(v.size):
        e = np.zeros_like(v)
        e[i] = h
        cols.append((fun(v + e) - fun(v - e)) / (2.0 * h))
    return np.column_stack(cols)


@pytest.mark.parametrize('h', [0.005, 1.0 / 150.0])     # the design's step, a 150 Hz sample
def test_jacobians_match_finite_differences(p, h):
    """F_k and G_k agree with central differences of report eq. 2 to 1e-6 (report Table 2)."""
    ekf = Ekf(p)
    F, G = (m.copy() for m in ekf.jacobians(X, U, h))
    assert np.allclose(F, _central_differences(lambda x: ekf.f(x, U, h), X), atol=1e-6)
    assert np.allclose(G, _central_differences(lambda u: ekf.f(X, u, h), U), atol=1e-6)


def test_ring_buffer_lookup_at_before_and_between_stamps():
    """The lookup returns the latest entry at or before the stamp, and None before the buffer."""
    buf = RingBuffer(depth=4)
    for k in range(6):                          # overfilled: stamps 0.010 to 0.025 remain
        buf.push(0.005 * k, np.full(5, float(k)))
    assert buf.at(0.020)[0] == 4                # at a stamp
    assert buf.at(0.0175)[0] == 3               # between two
    assert buf.at(0.010)[0] == 2                # at the oldest kept
    assert buf.at(1.0)[0] == 5                  # after the newest
    assert buf.at(0.009) is None                # before the oldest kept


def test_heading_innovation_wraps(p):
    """A fix at -179 deg against an estimate at +179 deg is a 2 deg innovation, not 358 deg."""
    ekf = Ekf(p)
    R = np.diag([0.004 ** 2, 0.004 ** 2, math.radians(0.5) ** 2])
    ekf.initialise(np.array([0.0, 0.0, math.radians(179.0)]), R, 0.0)
    for k in range(1, 11):
        ekf.predict(np.zeros(3), k * ekf.step)
    accepted, nis = ekf.update(np.array([0.0, 0.0, math.radians(-179.0)]), R, 5 * ekf.step)
    assert accepted and nis < ekf.gate
    assert abs(wrap_pi(ekf.x[2] - math.pi)) < math.radians(1.0)
    assert -math.pi < ekf.x[2] <= math.pi


def test_advance_follows_each_samples_interval(p):
    """Streams at 50 Hz with a lost sample, and at the IMU's uneven 150 Hz, match a steady one."""
    R = np.diag([1e-4, 1e-4, 1e-4])
    u = np.array([0.2, 0.0, 0.0])                   # constant forward acceleration
    ref, slow, uneven = Ekf(p), Ekf(p), Ekf(p)
    for ekf in (ref, slow, uneven):
        ekf.initialise(np.zeros(3), R, 0.0)
    for k in range(1, 201):                         # 1 s in the design's 5 ms steps
        assert ref.advance(u, k * ref.step) == 1
    stamps = [0.02 * k for k in range(1, 51) if k != 20]        # 50 Hz, one sample lost
    steps = [slow.advance(u, t) for t in stamps]
    assert steps[:3] == [4, 4, 4] and steps[19] == 8              # 4 a sample, 8 over the gap
    assert slow.advance(u, stamps[-1]) == 0                         # repeated stamp: no step
    assert sum(steps) == 200 and slow.t == pytest.approx(ref.t)
    assert np.allclose(slow.x, ref.x) and slow.x[3] == pytest.approx(0.2)

    # 150 Hz as measured on the robot: stamps on p4p_imu's 300 Hz poll grid, 1 to 3 polls apart
    polls = np.cumsum([2, 1, 2, 3, 2, 2, 1, 2] * 20)                # 300 polls, 1 s
    steps = [uneven.advance(u, n / 300.0) for n in polls]
    assert set(steps) == {1, 2} and uneven.t == pytest.approx(1.0)  # 2 only for the 10 ms gaps
    assert uneven.x[3] == pytest.approx(0.2)                        # velocity: exact at any step
    assert uneven.x[0] == pytest.approx(ref.x[0], abs=2e-4)         # position: Euler error, 0.1 mm


def test_floors_add_the_same_variance_per_second_at_any_step(p):
    """The floors add the same heading and velocity variance per second at any step size."""
    p['imu']['sigma_a'] = p['imu']['sigma_r_deg'] = 0.0
    R = np.diag([1e-4, 1e-4, 1e-4])
    at_200, at_150 = Ekf(p), Ekf(p)
    for ekf in (at_200, at_150):
        ekf.initialise(np.zeros(3), R, 0.0)
    for k in range(1, 201):
        at_200.advance(np.zeros(3), k / 200.0)
    for k in range(1, 151):
        at_150.advance(np.zeros(3), k / 150.0)
    cfg = p['ekf']
    per_second = 1.0 / cfg['step_s']
    assert at_200.P[2, 2] == pytest.approx(R[2, 2] + per_second * cfg['q_th'])
    assert at_200.P[3, 3] == pytest.approx(cfg['sigma_v0'] ** 2 + per_second * cfg['q_v'])
    assert at_150.P[2, 2] == pytest.approx(at_200.P[2, 2])
    assert at_150.P[3, 3] == pytest.approx(at_200.P[3, 3])


def test_a_late_fix_leaves_the_state_an_on_time_one_would(p):
    """A fix applied 147 ms after its photo leaves the state an on-time one would."""
    rng = np.random.default_rng(3)
    R = np.diag([0.004 ** 2, 0.004 ** 2, math.radians(0.5) ** 2])
    on_time, late = Ekf(p), Ekf(p)
    for ekf in (on_time, late):
        ekf.initialise(np.zeros(3), R, 0.0)
    capture, delivered = 60, 82                     # sample indices: the photo, and its pose
    z = None
    for k, u in enumerate(rng.normal(0.0, 0.3, (150, 3)), start=1):     # 1 s, moving
        for ekf in (on_time, late):
            ekf.advance(u, k / 150.0)
        if k == capture:
            z = on_time.x[:3] + np.array([0.003, -0.002, 0.004])
            assert on_time.update(z, R, capture / 150.0)[0]
        if k == delivered:
            assert late.update(z, R, capture / 150.0)[0]
    assert np.allclose(late.x, on_time.x, rtol=0.0, atol=1e-12)
    assert np.allclose(late.P, on_time.P, rtol=1e-9, atol=1e-15)


def test_a_re_anchor_keeps_the_turn_made_since_the_photo(p):
    """Restarting from a 100 ms old fix in a 0.5 rad/s turn replays the gyro since it."""
    R = np.diag([0.004 ** 2, 0.004 ** 2, math.radians(0.5) ** 2])
    ekf = Ekf(p)
    ekf.initialise(np.zeros(3), R, 0.0)
    for k in range(1, 61):                          # 0.4 s turning at 0.5 rad/s
        ekf.advance(np.array([0.0, 0.0, 0.5]), k / 150.0)
    z = np.array([-0.6, 0.015, math.radians(10.0)])
    ekf.reanchor(z, R, 45 / 150.0)                  # the photo, 15 steps (100 ms) ago
    assert ekf.x[0] == pytest.approx(z[0]) and ekf.x[1] == pytest.approx(z[1])
    assert ekf.x[2] == pytest.approx(z[2] + 0.5 * 15 / 150.0, abs=1e-9)
    assert np.allclose(ekf.x[3:], 0.0) and ekf.last_fix == 45 / 150.0
    assert ekf.P[0, 0] >= R[0, 0] and ekf.P[3, 3] >= p['ekf']['sigma_v0'] ** 2
    assert not ekf.update(z, R, 30 / 150.0)[0]      # older than the anchor: refused
    ekf.reanchor(z, R, -1.0)                        # before the buffer: initialised there
    assert ekf.x[2] == pytest.approx(z[2]) and ekf.t == -1.0


def test_a_fix_older_than_one_applied_is_rejected(p):
    """Replaying from an older fix would undo a newer correction, so it is refused."""
    R = np.diag([1e-4, 1e-4, 1e-4])
    ekf = Ekf(p)
    ekf.initialise(np.zeros(3), R, 0.0)
    for k in range(1, 31):
        ekf.advance(np.zeros(3), k / 150.0)
    assert ekf.update(np.zeros(3), R, 20 / 150.0)[0]
    assert not ekf.update(np.zeros(3), R, 10 / 150.0)[0]


@pytest.mark.parametrize('delay_s', [0.15, 0.3])
def test_precise_fixes_with_a_wifi_delay_do_not_oscillate(p, delay_s):
    """
    Robot still, sub-millimetre fixes 150 or 300 ms late, as on the robot over WiFi (2026-10-13).

    Shifting the later states by the photo-time correction, instead of replaying
    them, let each velocity correction misplace the present by itself times the
    delay. With fixes this precise that loop grew to 0.4 m/s and metres of error.
    """
    rng = np.random.default_rng(1)
    sa, sr = p['imu']['sigma_a'], math.radians(p['imu']['sigma_r_deg'])
    truth = np.array([-0.238, -0.096, math.radians(38.0)])
    sd = np.array([0.0003, 0.0003, math.radians(0.05)])
    R = np.diag(sd ** 2)
    ekf, pending, worst_v, worst_p = Ekf(p), [], 0.0, 0.0
    for k in range(1, 20 * 160 + 1):                # 20 s of IMU at 160 Hz
        t = k / 160.0
        if k % 11 == 0:                             # a photo at about 15 Hz
            pending.append((t + delay_s, truth + rng.normal(0.0, sd), t))
        if ekf.initialised:
            ekf.advance(rng.normal(0.0, [sa, sa, sr]), t)
        while pending and pending[0][0] <= t:
            _, z, capture = pending.pop(0)
            if ekf.initialised:
                ekf.update(z, R, capture)
            else:
                ekf.initialise(z, R, capture)
        if t > 10.0:
            worst_v = max(worst_v, abs(ekf.x[3]), abs(ekf.x[4]))
            worst_p = max(worst_p, abs(ekf.x[0] - truth[0]), abs(ekf.x[1] - truth[1]))
    assert worst_v < 0.02 and worst_p < 0.005


def test_a_still_robot_keeps_zero_velocity_through_a_late_fix(p):
    """While stopped every step ends at zero velocity, and a late fix's replay keeps it there."""
    R = np.diag([1e-6, 1e-6, 1e-6])
    ekf = Ekf(p)
    ekf.initialise(np.zeros(3), R, 0.0)
    push = np.array([0.4, -0.3, 0.0])                   # an IMU that says it is accelerating
    for k in range(1, 61):
        ekf.advance(push, k / 150.0, stopped=True)
    sigma_zv = p['ekf']['zupt_sigma_v']
    assert np.allclose(ekf.x[3:], 0.0) and ekf.P[3, 3] == pytest.approx(sigma_zv ** 2)
    assert ekf.update(np.array([0.002, 0.0, 0.0]), R, 20 / 150.0)[0]     # replayed 40 steps
    assert np.allclose(ekf.x[3:], 0.0)
    assert np.all(ekf.buffer.zv[ekf.buffer.t > 0])


def test_a_moved_board_moves_the_position_not_the_velocity_while_stopped(p):
    """Fixes stepping 5 cm while the robot stands: the pose follows, the velocity stays zero."""
    R = np.diag([0.002 ** 2, 0.002 ** 2, math.radians(0.3) ** 2])
    still, free = Ekf(p), Ekf(p)
    for ekf in (still, free):
        ekf.initialise(np.zeros(3), R, 0.0)
    peak = {id(still): 0.0, id(free): 0.0}
    for k in range(1, 151):             # 1 s; the board moves 5 cm in the first half
        t = k / 150.0
        z = np.array([min(t, 0.5) * 0.1, 0.0, 0.0])
        still.advance(np.zeros(3), t, stopped=True)
        free.advance(np.zeros(3), t)
        if k % 10 == 0:
            still.update(z, R, t)
            free.update(z, R, t)
        for ekf in (still, free):
            peak[id(ekf)] = max(peak[id(ekf)], float(np.hypot(*ekf.x[3:])))
    assert still.x[0] == pytest.approx(0.05, abs=0.005)
    assert peak[id(still)] == 0.0
    assert peak[id(free)] > 0.01        # without it, the board's motion became velocity
