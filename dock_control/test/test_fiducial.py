"""The fiducial chain: constants, sign tests (brief Section 2), a board turned in plane, R."""

import math

from dock_control.fiducial import FiducialChain
from dock_control.geometry import rot_z, se3, se3_inv, wrap_pi
import numpy as np
import pytest

# Straight in front, offset left, offset right, yawed 15 degrees.
FOUR_POSES = [(-0.5, 0.0, 0.0), (-0.5, 0.1, 0.0), (-0.5, -0.1, 0.0),
              (-0.5, 0.0, math.radians(15.0))]

# The pipeline's board frame in the normalised board frame (x along the outward
# normal, z up) for the board mounted the way it was marked "up": its x to the
# robot's right, its y down, so its z into the board.
R_UPRIGHT = np.array([[0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])


def _board_in_camera(chain, pose, spin=0.0):
    """
    Return T_OM, the board pose the pipeline reports with the robot at (x, y, theta).

    spin turns the board about its own normal (its z), about its centre.
    """
    x, y, theta = pose
    R = R_UPRIGHT @ rot_z(spin)
    T_T_board = chain.T_TM @ se3(R, -R @ chain.c_M)
    return chain.T_OB @ se3_inv(se3(rot_z(theta), [x, y, 0.0])) @ T_T_board


def _rotvec(phi):
    """Return the rotation matrix of a rotation vector (Rodrigues)."""
    angle = np.linalg.norm(phi)
    if angle == 0.0:
        return np.eye(3)
    k = phi / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K


@pytest.mark.parametrize('pose, tilt', [((-0.6, 0.08, math.radians(10.0)), 0.0),
                                        ((-0.6, 0.08, math.radians(10.0)), 0.15),
                                        ((-0.5, 0.0, 0.0), 0.0),       # the normal on atan2's cut
                                        ((-1.2, -0.2, math.radians(-25.0)), 0.1)])
def test_measurement_jacobian_matches_finite_differences(p, pose, tilt):
    """The Jacobian of linearise agrees with central differences of the chain, level or tilted."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, pose, spin=0.7)
    R0 = _rotvec(np.array([tilt, -0.5 * tilt, 0.0])) @ T_OM[:3, :3]     # roll and pitch too
    t0 = T_OM[:3, 3]
    z0, _, _, J = chain.linearise(R0, t0)

    def z_of(e):
        return np.array(chain.chain(_rotvec(e[3:]) @ R0, t0 + e[:3])[:3])

    h, I6 = 1e-5, np.eye(6)
    columns = []
    for i in range(6):
        d = z_of(h * I6[i]) - z_of(-h * I6[i])
        d[2] = wrap_pi(d[2])
        columns.append(d / (2.0 * h))
    assert np.allclose(z0, z_of(np.zeros(6)))
    assert np.allclose(J, np.column_stack(columns), atol=1e-6)


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
    """The chain inverts a known robot pose; level and upright, so roll and pitch are zero."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, pose)
    out = chain.chain(T_OM[:3, :3], T_OM[:3, 3])
    assert np.allclose(out[:3], pose, atol=1e-9)
    assert np.allclose(out[3:], 0.0, atol=1e-9)


@pytest.mark.parametrize('spin_deg', [0.0, 37.0, 90.0, 180.0, 270.0, -123.0])
@pytest.mark.parametrize('pose', FOUR_POSES + [(-1.5, 0.25, math.radians(-30.0))])
def test_the_boards_rotation_about_its_own_normal_is_normalised_out(p, pose, spin_deg):
    """A board turned in its own plane, about its centre, by any angle gives the same pose."""
    chain = FiducialChain(p)
    T_OM = _board_in_camera(chain, pose, math.radians(spin_deg))
    x, y, theta, roll, pitch = chain.chain(T_OM[:3, :3], T_OM[:3, 3])
    assert np.allclose((x, y, theta), pose, atol=1e-9)
    assert pitch == pytest.approx(0.0, abs=1e-9)
    # roll reports the turn, counter-clockwise as the robot sees it (the board's z points away)
    assert wrap_pi(roll + math.radians(spin_deg)) == pytest.approx(0.0, abs=1e-9)


def _seen(chain, ahead, right):
    """Return the board pose with its centre `ahead` of the lens and `right` of it, upright."""
    c = np.array((right, 0.0, ahead))           # optical: x right, y down, z forward
    return np.eye(3), c - chain.c_M


def test_signs_by_hand(p):
    """Board poses written down by hand, not through the chain's own constants."""
    chain = FiducialChain(p)
    lens = p['geom']['cam_xyz'][0]
    docked = p['geom']['marker_m'][0] - lens       # the board centre from the lens when docked
    # docked, facing the board: z = 0
    assert np.allclose(chain.chain(*_seen(chain, docked, 0.0))[:3], 0.0, atol=1e-12)
    # 1 m further back: x = -1
    assert np.allclose(chain.chain(*_seen(chain, docked + 1.0, 0.0))[:3], (-1.0, 0.0, 0.0),
                       atol=1e-12)
    # the robot 0.1 m to the left, so the board is 0.1 m to its right: y = +0.1
    assert np.allclose(chain.chain(*_seen(chain, docked, 0.1))[:3], (0.0, 0.1, 0.0), atol=1e-12)
    # the board's normal turned by a about the vertical, as the robot sees it after turning
    # counter-clockwise by a: theta = +a
    a = math.radians(20.0)
    R_OM = _rotvec(np.array((0.0, a, 0.0)))      # about optical y, which points down
    T_OM = se3(R_OM, np.zeros(3))
    T_OM[:3, 3] = np.array((0.0, 0.0, 1.0)) - R_OM @ chain.c_M
    z = chain.chain(T_OM[:3, :3], T_OM[:3, 3])[:3]
    assert z[2] == pytest.approx(a, abs=1e-12)
