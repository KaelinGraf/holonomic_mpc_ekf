"""Mecanum kinematics against report eq. 18 and Table 3."""

from dock_control.kinematics import fk_matrix, ik_matrix
import numpy as np


def _matrices(p):
    g = p['geom']
    return ik_matrix(g['Rw'], g['lx'], g['ly']), fk_matrix(g['Rw'], g['lx'], g['ly'])


def test_forward_inverts_inverse(p):
    """Check that J^+ J^- = I3."""
    J_inv, J_plus = _matrices(p)
    assert J_inv.shape == (4, 3) and J_plus.shape == (3, 4)
    assert np.allclose(J_plus @ J_inv, np.eye(3), atol=1e-12)


def test_null_vector_only_scrubs_the_rollers(p):
    """Check that J^+ [1, 1, -1, -1]' = 0: that wheel combination moves nothing."""
    _, J_plus = _matrices(p)
    assert np.allclose(J_plus @ [1.0, 1.0, -1.0, -1.0], 0.0, atol=1e-12)


def test_wheel_signs(p):
    """Check that forward, left and yaw turn FL, FR, RL, RR with the signs of report eq. 18."""
    J_inv, _ = _matrices(p)
    assert np.array_equal(np.sign(J_inv @ [1.0, 0.0, 0.0]), [1, 1, 1, 1])
    assert np.array_equal(np.sign(J_inv @ [0.0, 1.0, 0.0]), [-1, 1, 1, -1])
    assert np.array_equal(np.sign(J_inv @ [0.0, 0.0, 1.0]), [-1, 1, -1, 1])
