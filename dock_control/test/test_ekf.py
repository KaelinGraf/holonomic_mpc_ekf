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


def test_jacobians_match_finite_differences(p):
    """F_k and G_k agree with central differences of report eq. 2 to 1e-6 (report Table 2)."""
    ekf = Ekf(p)
    F, G = (m.copy() for m in ekf.jacobians(X, U))
    assert np.allclose(F, _central_differences(lambda x: ekf.f(x, U), X), atol=1e-6)
    assert np.allclose(G, _central_differences(lambda u: ekf.f(X, u), U), atol=1e-6)


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
        ekf.predict(np.zeros(3), k * ekf.T)
    accepted, nis = ekf.update(np.array([0.0, 0.0, math.radians(-179.0)]), R, 5 * ekf.T)
    assert accepted and nis < ekf.gate
    assert abs(wrap_pi(ekf.x[2] - math.pi)) < math.radians(1.0)
    assert -math.pi < ekf.x[2] <= math.pi


def test_advance_covers_the_elapsed_time_at_any_imu_rate(p):
    """A 50 Hz stream, a dropped sample and a repeated stamp step the same time as 200 Hz."""
    R = np.diag([1e-4, 1e-4, 1e-4])
    u = np.array([0.2, 0.0, 0.0])                   # constant forward acceleration
    fast, slow = Ekf(p), Ekf(p)
    for ekf in (fast, slow):
        ekf.initialise(np.zeros(3), R, 0.0)
    T = fast.T
    for k in range(1, 201):                         # 1 s at 200 Hz
        assert fast.advance(u, k * T) == 1
    stamps = [0.02 * k for k in range(1, 51) if k != 20]        # 50 Hz, one sample lost
    steps = [slow.advance(u, t) for t in stamps]
    assert steps[:3] == [4, 4, 4] and steps[19] == 8              # 4 a sample, 8 over the gap
    assert slow.advance(u, stamps[-1]) == 0                         # repeated stamp: no step
    assert sum(steps) == 200 and slow.t == pytest.approx(fast.t)
    assert np.allclose(slow.x, fast.x) and slow.x[3] == pytest.approx(0.2)


def test_correction_reaches_later_buffer_entries():
    """A correction shifts the entries at and after the capture stamp, not the ones before."""
    buf = RingBuffer(depth=4)
    for k in range(4):
        buf.push(0.005 * k, np.zeros(5))
    buf.correct_from(0.010, np.ones(5))
    assert buf.at(0.005)[0] == 0.0
    assert buf.at(0.010)[0] == 1.0 and buf.at(0.015)[0] == 1.0
