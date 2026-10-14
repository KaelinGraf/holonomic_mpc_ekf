"""
Fiducial pre-transform (brief Section 2): board pose to the pseudo-measurement z.

A library, not a node: ekf_node calls it in its inference_result callback, so a
pose costs no extra hop. The input is RosInferenceResult.pose from the
Conv-ChArT pipeline: the board in the camera's optical frame (frame_id
pi_camera), from OpenCV's IPPE solve.

The board's rotation about its own normal (its z axis) is normalised out: only
the centre's position and the normal's yaw are used. Both are carried into
base_link and laid onto the ground plane, which gives a normalised board frame:
at the board centre c (2-D), x along the outward normal's horizontal part (yaw
phi), z up. That frame is the planar marker of report eq. 5, so
theta = psi_0 - phi and p = m - R(theta) c place the robot in the target
frame. A board turned in its own plane by any angle, and whichever lattice
corner the pipeline numbers first, gives the same pose. m and psi_0 place the
normalised board frame in the target frame: m the board centre, psi_0 = 180 deg
its normal pointing back at the docking robot.

Each pose arrives with its own covariance, measured live by the pipeline, and
measure() carries it through the chain's Jacobian into R for z, so nothing about
the camera's noise is fixed at start-up. The range model sigma_p(d), sigma_th is
kept only for simulating fixes.
"""

import math

from dock_control.geometry import rot_z, se3, se3_inv, wrap_pi
import numpy as np

R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])

JACOBIAN_STEP = 1e-6


def _skew(v):
    """Return [v]x, the matrix with [v]x w = v x w."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotation(phi):
    """Return the rotation matrix of a rotation vector (Rodrigues)."""
    angle = float(np.linalg.norm(phi))
    if angle == 0.0:
        return np.eye(3)
    K = _skew(phi / angle)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K


class FiducialChain:
    """The constant factors of the chain, the per-frame step, and the simulation noise model."""

    def __init__(self, p):
        g, fid = p['geom'], p['fid']

        squares, square = g['board_squares'], g['board_square_m']
        self.c_M = np.array((0.5 * squares[0] * square, 0.5 * squares[1] * square, 0.0))
        self.m = np.array(g['marker_m'][:2], dtype=float)
        self.psi0 = math.radians(g['marker_psi0_deg'])
        self.T_TM = se3(rot_z(self.psi0), [self.m[0], self.m[1], 0.0])
        self.R_BO = R_LINK_OPTICAL
        self.p_BO = np.array(g['cam_xyz'], dtype=float)
        self.T_OB = se3_inv(se3(self.R_BO, self.p_BO))
        h = JACOBIAN_STEP
        self._dt = [s * h * e for e in np.eye(3) for s in (1.0, -1.0)]
        self._dR = [_rotation(s * h * e) for e in np.eye(3) for s in (1.0, -1.0)]
        self._J = np.zeros((3, 6))
        self.sigma_p0 = fid['sigma_p0']
        self.inv_d0_sq = 1.0 / fid['d0'] ** 2
        self.sigma_th = math.radians(fid['sigma_th_deg'])
        floor = fid.get('sigma_th_floor_deg')
        self.var_th_floor = math.radians(floor) ** 2 if floor else 0.0

    def chain(self, R_OM, t_OM):
        """Return (x, y, theta, roll, pitch) of base_link in the target frame."""
        c = self.R_BO @ (R_OM @ self.c_M + t_OM) + self.p_BO
        n = -(self.R_BO @ R_OM[:, 2])
        horizontal = math.hypot(n[0], n[1])
        theta = wrap_pi(self.psi0 - math.atan2(n[1], n[0]))
        ct, st = math.cos(theta), math.sin(theta)
        x = self.m[0] - (ct * c[0] - st * c[1])
        y = self.m[1] - (st * c[0] + ct * c[1])
        pitch = math.atan2(n[2], horizontal)
        roll = 0.0
        if horizontal > 1e-9:
            right = np.array((-n[1], n[0], 0.0)) / horizontal
            up = np.cross(n, right)
            bx = self.R_BO @ R_OM[:, 0]
            roll = math.atan2(bx @ up, bx @ right)
        return x, y, theta, roll, pitch

    def linearise(self, R_OM, t_OM):
        """Return (z, roll, pitch, J): the chain at one board pose and its 3x6 Jacobian."""
        x, y, theta, roll, pitch = self.chain(R_OM, t_OM)
        J, inv_2h = self._J, 0.5 / JACOBIAN_STEP
        for i in range(3):
            a = self.chain(R_OM, t_OM + self._dt[2 * i])
            b = self.chain(R_OM, t_OM + self._dt[2 * i + 1])
            J[0, i], J[1, i] = (a[0] - b[0]) * inv_2h, (a[1] - b[1]) * inv_2h
            J[2, i] = wrap_pi(a[2] - b[2]) * inv_2h
            a = self.chain(self._dR[2 * i] @ R_OM, t_OM)
            b = self.chain(self._dR[2 * i + 1] @ R_OM, t_OM)
            J[0, 3 + i], J[1, 3 + i] = (a[0] - b[0]) * inv_2h, (a[1] - b[1]) * inv_2h
            J[2, 3 + i] = wrap_pi(a[2] - b[2]) * inv_2h
        return np.array((x, y, theta)), roll, pitch, J.copy()

    def measure(self, R_OM, t_OM, pose_cov):
        """Return (z, R, roll, pitch) from one board pose and its live 6x6 covariance."""
        z, roll, pitch, J = self.linearise(R_OM, t_OM)
        R = J @ pose_cov @ J.T
        R[2, 2] = max(R[2, 2], self.var_th_floor)
        return z, R, roll, pitch

    def sigma_p(self, d):
        """Return the simulated position standard deviation at range d (report Section 2.2)."""
        return self.sigma_p0 * math.sqrt(1.0 + d * d * self.inv_d0_sq)
