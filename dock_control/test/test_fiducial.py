"""The fiducial chain: its constants, the four-pose sign test (brief Section 2), the live R."""

import math

from dock_control.fiducial import FiducialChain
from dock_control.geometry import rot_z, se3, se3_inv
import numpy as np
import pytest

# Straight in front, offset left, offset right, yawed 15 degrees.
FOUR_POSES = [(-0.5, 0.0, 0.0), (-0.5, 0.1, 0.0), (-0.5, -0.1, 0.0),
              (-0.5, 0.0, math.radians(15.0))]


def _board_in_camera(chain, pose):
    """Return T_OM, the board pose the pipeline reports with the robot at (x, y, theta)."""
    x, y, theta = pose
    return chain.T_OB @ se3_inv(se3(rot_z(theta), [x, y, 0.0])) @ chain.T_TM


def _rotvec(phi):
    """Return the rotation matrix of a rotation vector (Rodrigues)."""
    angle = np.linalg.norm(phi)
    if angle == 0.0:
        return np.eye(3)
    k = phi / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K


@pytest.mark.parametrize('tilt', [0.0, 0.15])
def test_measurement_jacobian_matches_finite_differences(p, tilt):
    """The Jacobian of linearise agrees with central differences of the chain, level or tilted."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, (-0.6, 0.08, math.radians(10.0)))
    R0 = _rotvec(np.array([tilt, -0.5 * tilt, 0.0])) @ T_OM[:3, :3]     # roll and pitch too
    t0 = T_OM[:3, 3]
    z0, _, _, J = chain.linearise(R0, t0)

    def z_of(e):
        return np.array(chain.chain(_rotvec(e[3:]) @ R0, t0 + e[:3])[:3])

    h, I6 = 1e-6, np.eye(6)
    J_fd = np.column_stack([(z_of(h * I6[i]) - z_of(-h * I6[i])) / (2.0 * h) for i in range(6)])
    assert np.allclose(z0, z_of(np.zeros(6)))
    assert np.allclose(J, J_fd, atol=1e-6)


def test_pose_covariance_reaches_z_through_the_chain(p):
    """Square on, the camera's depth noise becomes z's x noise and its sideways noise z's y."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, (-0.5, 0.0, 0.0))
    depth = np.diag([0.0, 0.0, 1e-4, 0.0, 0.0, 0.0])     # along the optical axis
    side = np.diag([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])      # image right
    _, R_depth, _, _ = chain.measure(T_OM[:3, :3], T_OM[:3, 3], depth)
    _, R_side, _, _ = chain.measure(T_OM[:3, :3], T_OM[:3, 3], side)
    assert np.allclose(R_depth, np.diag([1e-4, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(R_side, np.diag([0.0, 1e-4, 0.0]), atol=1e-12)


def test_optical_axes(p):
    """Optical z looks forward, image right is the robot's right, and image down is down."""
    R_BO = se3_inv(FiducialChain(p).T_OB)[:3, :3]
    assert np.allclose(R_BO @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert np.allclose(R_BO @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    assert np.allclose(R_BO @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0])


def test_board_seen_square_on_is_the_identity(p):
    """Facing the board squarely gives R = I, as OpenCV reports a board facing the camera."""
    T_OM = _board_in_camera(FiducialChain(p), (-0.5, 0.0, 0.0))
    assert np.allclose(T_OM[:3, :3], np.eye(3))


@pytest.mark.parametrize('pose', FOUR_POSES)
def test_chain_recovers_the_four_poses(p, pose):
    """The chain inverts a known robot pose; level, so roll and pitch are zero."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, pose)
    out = chain.chain(T_OM[:3, :3], T_OM[:3, 3])
    assert np.allclose(out[:3], pose, atol=1e-9)
    assert np.allclose(out[3:], 0.0, atol=1e-9)
