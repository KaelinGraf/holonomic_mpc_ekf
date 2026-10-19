"""
Mecanum kinematics (report eq. 18): wheel order FL, FR, RL, RR, rollers at 45 deg.

Both matrices depend only on the geometry, so they are built once at start-up:
J^- gives the MPC its wheel-speed constraint rows, which are therefore constant
too. The firmware does its own kinematics for the wheels, so R_w, l_x + l_y and
w_max here must match its config.h (WHEEL_MAX_MPS, CHASSIS_L_PLUS_W).
"""

import numpy as np


def ik_matrix(Rw, lx, ly):
    """Return J^- (4x3): wheel speeds omega = J^- u from the body velocity u = [u_b, v_b, r_b].

    Args:
        Rw: Wheel radius [m].
        lx: Half the distance between the front and rear wheel centres [m].
        ly: Half the distance between the left and right wheel centres [m].
    """
    L = lx + ly
    return np.array([[1.0, -1.0, -L],
                     [1.0, 1.0, L],
                     [1.0, 1.0, -L],
                     [1.0, -1.0, L]]) / Rw


def fk_matrix(Rw, lx, ly):
    """Return J^+ (3x4): the body velocity u = J^+ omega from the wheel speeds.

    Args:
        Rw: Wheel radius [m].
        lx: Half the distance between the front and rear wheel centres [m].
        ly: Half the distance between the left and right wheel centres [m].
    """
    k = 1.0 / (lx + ly)
    return 0.25 * Rw * np.array([[1.0, 1.0, 1.0, 1.0],
                                 [-1.0, 1.0, 1.0, -1.0],
                                 [-k, k, -k, k]])
