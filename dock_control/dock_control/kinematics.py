"""
Mecanum kinematics (report eq. 18): wheel order FL, FR, RL, RR, rollers at 45 deg.

Both matrices depend only on the geometry, so they are built once at start-up:
J^- gives the MPC its wheel-speed constraint rows, which are therefore constant
too. The firmware does its own kinematics for the wheels, so R_w, l_x + l_y and
w_max here must match its config.h (WHEEL_MAX_MPS, CHASSIS_L_PLUS_W).
"""

import numpy as np


def ik_matrix(Rw, lx, ly):
    """
    Return J^- (4x3): wheel speeds omega = J^- u from the body velocity u = [u_b, v_b, r_b].

    L = lx + ly; mecanum_ik.m is the reference.
    """
    L = lx + ly
    return np.array([[1.0, -1.0, -L],
                     [1.0, 1.0, L],
                     [1.0, 1.0, -L],
                     [1.0, -1.0, L]]) / Rw


def fk_matrix(Rw, lx, ly):
    """
    Return J^+ (3x4): the body velocity u = J^+ omega from the wheel speeds.

    J^+ J^- = I3, and J^+ [1, 1, -1, -1]' = 0 (the combination that only scrubs
    the rollers); mecanum_fk.m is the reference.
    """
    k = 1.0 / (lx + ly)
    return 0.25 * Rw * np.array([[1.0, 1.0, 1.0, 1.0],
                                 [-1.0, 1.0, 1.0, -1.0],
                                 [-k, k, -k, k]])
