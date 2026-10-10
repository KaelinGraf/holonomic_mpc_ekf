"""Rigid-body helpers: angle wrapping, rigid inverses and quaternions."""

import math

from dock_control.geometry import quat_to_rot, rot_to_quat, rot_z, se3, se3_inv, wrap_pi
import numpy as np
import pytest


def _rot(rx, ry, rz):
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rot_z(rz) @ Ry @ Rx


def test_wrap_pi_lands_in_the_half_open_interval():
    """Every angle maps into (-pi, pi], with pi kept and -pi sent to pi."""
    assert wrap_pi(math.pi) == pytest.approx(math.pi)
    assert wrap_pi(-math.pi) == pytest.approx(math.pi)
    assert wrap_pi(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_pi(-0.1) == pytest.approx(-0.1)
    a = np.linspace(-20.0, 20.0, 4001)
    w = wrap_pi(a)
    assert np.all(w > -math.pi) and np.all(w <= math.pi)
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))


def test_rigid_inverse():
    """The transpose-based inverse undoes the transform."""
    T = se3(rot_z(0.7), [0.1, -0.2, 0.3])
    assert np.allclose(se3_inv(T) @ T, np.eye(4))


@pytest.mark.parametrize('angles', [(0.3, -0.2, 1.1), (math.pi, 0.0, 0.0), (0.0, math.pi, 0.0),
                                    (0.0, 0.0, math.pi), (2.0, 2.5, -3.0)])
def test_quaternion_round_trip(angles):
    """The two quaternion conversions are inverses, including at 180 degrees."""
    R = _rot(*angles)
    q = rot_to_quat(R)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert np.allclose(quat_to_rot(*q), R)
