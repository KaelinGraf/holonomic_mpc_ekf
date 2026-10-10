"""The fiducial chain: its constant transforms, and the four-pose sign test of brief Section 2."""

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
