"""
Fiducial pre-transform (brief Section 2): board pose to the pseudo-measurement z.

A library, not a node: ekf_node calls it in its inference_result callback, so a
pose costs no extra hop. The input is RosInferenceResult.pose from the
Conv-ChArT pipeline: the board in the camera's optical frame (frame_id
pi_camera), from OpenCV's IPPE solve.

The chain is T_TB = T_TM inv(T_CM) inv(T_BC). Two of its three factors never
change, so FiducialChain builds them once, with every constant axis change
folded in. Per frame, only inv(T_CM) is formed (a transpose, not an inverse) and
multiplied in.

Each pose arrives with its own covariance, measured live by the pipeline, and
measure() carries it through the same chain into R for z, so nothing about the
camera's noise is fixed at start-up. The range model sigma_p(d), sigma_th is
kept only for simulating fixes.
"""

import math

from dock_control.geometry import rot_z, se3, se3_inv
import numpy as np

R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])

R_MARKER_BOARD = np.array([[0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


def _skew(v):
    """Return [v]x, the matrix with [v]x w = v x w."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class FiducialChain:
    """The constant factors of the chain, the per-frame step, and the simulation noise model."""

    def __init__(self, p):
        g, fid = p['geom'], p['fid']

        T_T_marker = se3(rot_z(math.radians(g['marker_psi0_deg'])),
                         [g['marker_m'][0], g['marker_m'][1], 0.0])
        self.T_TM = T_T_marker @ se3(R_MARKER_BOARD, np.zeros(3))
        self.T_OB = se3_inv(se3(R_LINK_OPTICAL, g['cam_xyz']))
        self.R_TM, self.p_TM = self.T_TM[:3, :3].copy(), self.T_TM[:3, 3].copy()
        self.R_OB, self.p_OB = self.T_OB[:3, :3].copy(), self.T_OB[:3, 3].copy()
        self.sigma_p0 = fid['sigma_p0']
        self.inv_d0_sq = 1.0 / fid['d0'] ** 2
        self.sigma_th = math.radians(fid['sigma_th_deg'])

    def chain(self, R_OM, t_OM):
        """Return (x, y, theta, roll, pitch) of base_link in the target frame."""
        R_MO = R_OM.T
        R = self.R_TM @ R_MO @ self.R_OB
        p = self.R_TM @ (R_MO @ (self.p_OB - t_OM)) + self.p_TM
        return (p[0], p[1], math.atan2(R[1, 0], R[0, 0]),
                math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1.0, min(1.0, R[2, 0]))))

    def linearise(self, R_OM, t_OM):
        """Return (z, roll, pitch, J): the chain at one board pose and its 3x6 Jacobian."""
        M = self.R_TM @ R_OM.T
        R = M @ self.R_OB
        v = self.p_OB - t_OM
        p = M @ v + self.p_TM
        c2 = R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]
        w = np.array((-R[2, 0] * R[0, 0] / c2, -R[2, 0] * R[1, 0] / c2, 1.0))
        J = np.zeros((3, 6))
        J[:2, :3] = -M[:2]
        J[:2, 3:] = (M @ _skew(v))[:2]
        J[2, 3:] = -w @ M
        z = np.array((p[0], p[1], math.atan2(R[1, 0], R[0, 0])))
        return (z, math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1.0, min(1.0, R[2, 0]))), J)

    def measure(self, R_OM, t_OM, pose_cov):
        """Return (z, R, roll, pitch) from one board pose and its live 6x6 covariance."""
        z, roll, pitch, J = self.linearise(R_OM, t_OM)
        return z, J @ pose_cov @ J.T, roll, pitch

    def sigma_p(self, d):
        """Return the simulated position standard deviation at range d (report Section 2.2)."""
        return self.sigma_p0 * math.sqrt(1.0 + d * d * self.inv_d0_sq)
