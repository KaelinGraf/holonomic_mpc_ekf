"""
Rigid-body helpers shared by the fiducial chain, the EKF and the MPC.

Anything built from these that does not change at run time (the static
transforms and their inverses) is built once in a constructor, never per sample.
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi
GRAVITY = 9.80665


def wrap_pi(a):
    """Wrap an angle, or an array of angles, into (-pi, pi].

    Args:
        a: Angle or array of angles [rad].
    """
    return math.pi - (math.pi - a) % TWO_PI


def rot2(theta):
    """Return the planar rotation R(theta) of report eq. 1.

    Args:
        theta: Rotation angle [rad].
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def rot_z(psi):
    """Return the 3-D rotation by psi about z.

    Args:
        psi: Rotation angle about z [rad].
    """
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def se3(R, t):
    """Return the 4x4 homogeneous transform with rotation R and translation t.

    Args:
        R: Rotation matrix (3x3).
        t: Translation (3-vector).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T):
    """Invert a rigid transform with a transpose rather than a general inverse.

    Args:
        T: Rigid transform (4x4).
    """
    Rt = T[:3, :3].T
    Ti = np.eye(4)
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ T[:3, 3]
    return Ti


def body_gravity_xy(x, y, z, w):
    """Return the x and y an accelerometer at rest reads from gravity [m/s^2].

    Only roll and pitch enter, so a drifting yaw does not matter.

    Args:
        x: Quaternion x of the orientation, body to world with z up.
        y: Quaternion y.
        z: Quaternion z.
        w: Quaternion w.
    """
    return 2.0 * GRAVITY * (x * z - w * y), 2.0 * GRAVITY * (y * z + w * x)


def quat_to_rot(x, y, z, w):
    """Return the rotation matrix of a unit quaternion, in ROS order (x, y, z, w).

    Args:
        x: Quaternion x.
        y: Quaternion y.
        z: Quaternion z.
        w: Quaternion w.
    """
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                     [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                     [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]])


def rot_to_quat(R):
    """Return the unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method).

    Args:
        R: Rotation matrix (3x3).
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
            0.25 * s, (R[1, 0] - R[0, 1]) / s)
