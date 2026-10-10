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
"""

import math

from dock_control.geometry import rot_z, se3, se3_inv
import numpy as np

# camera_link <- optical frame (pi_camera). Optical: z forward, x right, y down;
# camera_link: x forward, y left, z up.
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])

# planar marker <- pipeline board. The pipeline's object points are
# canon_lattice() * square_length with z = 0 (Conv-ChArT-Wireless-Inference
# src/inference.py): x along the corner columns, y down the rows, so z points
# into the board. The planar marker frame of the report has x along the face's
# outward normal (psi_0 = 180 deg points it back at the robot) and z up. This
# assumes corner 0 is top-left seen from the front: verify it with the four-pose
# sign test (brief Section 10.b, step 4) before the EKF is switched on.
R_MARKER_BOARD = np.array([[0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


class FiducialChain:
    """The constant factors of the chain, the per-frame step, and the noise model."""

    def __init__(self, p):
        g, fid = p['geom'], p['fid']

        # --- constant, built once ---
        # target <- board: the planar marker at m (2-D) with yaw psi_0, both
        # measured in T, then the board axes. m locates the board frame's origin,
        # which is the lattice's outer corner (one square beyond corner 0), not
        # the board centre.
        T_T_marker = se3(rot_z(math.radians(g['marker_psi0_deg'])),
                         [g['marker_m'][0], g['marker_m'][1], 0.0])
        self.T_TM = T_T_marker @ se3(R_MARKER_BOARD, np.zeros(3))
        # optical <- base: the inverse of base <- camera_link <- optical (the
        # camera is mounted aligned with the body at cam_xyz), so no static
        # transform is inverted per frame.
        self.T_OB = se3_inv(se3(R_LINK_OPTICAL, g['cam_xyz']))
        # The same two, split into rotation and translation for chain().
        self.R_TM, self.p_TM = self.T_TM[:3, :3].copy(), self.T_TM[:3, 3].copy()
        self.R_OB, self.p_OB = self.T_OB[:3, :3].copy(), self.T_OB[:3, 3].copy()
        # noise model: sigma_p(d) = sigma_p0 sqrt(1 + d^2 / d0^2); sigma_th is constant
        self.sigma_p0 = fid['sigma_p0']
        self.inv_d0_sq = 1.0 / fid['d0'] ** 2
        self.sigma_th = math.radians(fid['sigma_th_deg'])

    def chain(self, R_OM, t_OM):
        """
        Return (x, y, theta, roll, pitch) of base_link in the target frame.

        R_OM, t_OM: the board pose in the optical frame, as the pipeline returns
        it (geometry.quat_to_rot turns its quaternion into R_OM).

        T_TB = T_TM inv(T_OM) T_OB, multiplied out so that nothing is inverted:
        R_TB = R_TM R_OM' R_OB and p_TB = R_TM R_OM' (p_OB - t_OM) + p_TM. Yaw,
        pitch and roll are the Z-Y-X angles of R_TB; roll and pitch are the
        monitored channels. With the camera level this is the planar form of
        report eq. 5: theta_z = psi_0 - psi, p_z = m - R(theta_z)(t + l_c).
        """
        R_MO = R_OM.T
        R = self.R_TM @ R_MO @ self.R_OB
        p = self.R_TM @ (R_MO @ (self.p_OB - t_OM)) + self.p_TM
        return (p[0], p[1], math.atan2(R[1, 0], R[0, 0]),
                math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1.0, min(1.0, R[2, 0]))))

    def sigma_p(self, d):
        """Return the position standard deviation at range d (report Section 2.2)."""
        return self.sigma_p0 * math.sqrt(1.0 + d * d * self.inv_d0_sq)
