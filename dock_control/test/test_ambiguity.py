"""
IPPE ambiguity in ekf_node: the solution with the smaller NIS is fused; one that agrees may start.

resolve() on its own, then ekf_node.on_pose with real board poses through the chain. Needs the
messages ekf_node imports (rclpy, dock_msgs, convchart_interfaces); skipped without them. The
node runs in a private domain, discovery limited to this machine, and is never spun.
"""

import math
import os

from dock_control.ekf import Ekf
from dock_control.fiducial import _rotation, FiducialChain
from dock_control.geometry import rot_to_quat, rot_z, se3, se3_inv
import numpy as np
import pytest

ekf_node = pytest.importorskip('dock_control.ekf_node')

R = np.diag([0.002 ** 2, 0.002 ** 2, math.radians(0.3) ** 2])
DOMAIN_ID = 111 + os.getpid() % 20     # never the robot's 42
# the pipeline's board frame in the normalised board frame, mounted as marked "up"
R_UPRIGHT = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _fix(x, y, theta_deg):
    return (np.array([x, y, math.radians(theta_deg)]), R.copy(), 0.0, 0.0, 1.0)


def _running(p):
    ekf = Ekf(p)
    ekf.initialise(np.array([-0.5, 0.0, 0.0]), R, 0.0)
    for k in range(1, 16):
        ekf.advance(np.zeros(3), k / 150.0)
    return ekf


def test_the_solution_with_the_smaller_nis_is_fused_and_still_gated(p):
    """Whichever comes first, the one nearer the estimate is chosen; update gates the far one."""
    ekf, t = _running(p), 10 / 150.0
    near, far = _fix(-0.5, 0.001, 0.2), _fix(-0.5, 0.0, 6.0)
    chosen, valid, _ = ekf_node.resolve(ekf, [(far, True), (near, False)], True, t, True)
    assert chosen[0] is near[0] and valid is False
    chosen, valid, _ = ekf_node.resolve(ekf, [(near, True), (far, False)], True, t, True)
    assert chosen[0] is near[0] and valid is True
    assert not ekf.update(far[0], far[1], t)[0]     # NIS (6 / 0.3)^2 = 400, over the gate
    assert ekf.update(near[0], near[1], t)[0]


def test_an_ambiguous_pose_starts_the_filter_from_its_first_solution(p):
    """Agreeing or not, the first solution starts it, wide where the two disagree."""
    ekf = Ekf(p)
    first = _fix(-0.8, 0.05, 3.0)
    disagrees = _fix(-0.8, 0.12, 9.0)
    for estimating in (True, False):
        chosen, _, may_start = ekf_node.resolve(ekf, [(first, True), (disagrees, True)], True,
                                                1.0, estimating)
        assert chosen[0] is first[0] and may_start
        assert chosen[1][1, 1] > R[1, 1] + 0.06 ** 2 and chosen[1][2, 2] > math.radians(5.9) ** 2
    assert ekf_node.resolve(ekf, [(first, True)], True, 1.0, True)[2]
    assert ekf_node.resolve(ekf, [(first, True)], False, 1.0, True)[2]
    assert ekf_node.planar_agree(_fix(0.0, 0.0, 179.9), _fix(0.0, 0.0, -179.9))   # wrapped


@pytest.fixture
def node():
    rclpy = pytest.importorskip('rclpy')
    saved = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    rclpy.init(domain_id=DOMAIN_ID)
    n = ekf_node.EkfNode()
    n.fixes = []
    n.publish_fix = lambda stamp, fix, cov_valid, ambiguous: n.fixes.append((fix, ambiguous))
    try:
        yield n
    finally:
        n.destroy_node()
        rclpy.shutdown()
        if saved is None:
            os.environ.pop('ROS_AUTOMATIC_DISCOVERY_RANGE', None)
        else:
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = saved


def _pose(T_OM, msg_pose):
    q = rot_to_quat(T_OM[:3, :3])
    msg_pose.pose.position.x, msg_pose.pose.position.y, msg_pose.pose.position.z = T_OM[:3, 3]
    o = msg_pose.pose.orientation
    o.x, o.y, o.z, o.w = q
    msg_pose.covariance = list(np.diag([1e-7, 1e-7, 1e-6, 1e-5, 1e-5, 1e-5]).ravel())


def _ambiguous_result(node, turn_alt):
    """Return an ambiguous result 0.6 m back, its second solution turned by turn_alt (3-vector)."""
    from convchart_interfaces.msg import RosInferenceResult
    chain = node.chain
    T_T_board = chain.T_TM @ se3(R_UPRIGHT, -R_UPRIGHT @ chain.c_M)
    T_OM = chain.T_OB @ se3_inv(se3(rot_z(0.0), [-0.6, 0.0, 0.0])) @ T_T_board
    centre = T_OM[:3, :3] @ chain.c_M + T_OM[:3, 3]
    R_alt = _rotation(np.array(turn_alt))
    T_alt = np.eye(4)
    T_alt[:3, :3] = R_alt @ T_OM[:3, :3]            # turned about the board centre
    T_alt[:3, 3] = centre - T_alt[:3, :3] @ chain.c_M
    msg = RosInferenceResult(ambiguous=True, covariance_valid=True, covariance_valid_alt=True)
    msg.header.stamp.sec = 100
    _pose(T_OM, msg.pose)
    _pose(T_alt, msg.pose_alt)
    return msg


def test_on_pose_publishes_an_ambiguous_fix_when_idle_and_starts_from_the_first(node):
    """
    Publish an ambiguous fix when idle; once armed, start from the first solution.

    Turned 8 deg about the vertical, the two solutions disagree on theta, and the
    filter still starts (from the first), wide in theta where they disagree.
    """
    assert isinstance(node.chain, FiducialChain)
    node.on_pose(_ambiguous_result(node, (0.0, 0.0, 0.0)))          # idle: published, not fused
    assert len(node.fixes) == 1 and node.fixes[0][1] and not node.ekf.initialised
    node.active = node.zeroed = True                                 # armed, zeroing done
    node.on_pose(_ambiguous_result(node, (0.0, math.radians(8.0), 0.0)))   # about optical y
    assert node.ekf.initialised and len(node.fixes) == 2
    assert node.ekf.P[2, 2] > math.radians(1.0) ** 2                 # their disagreement


def test_an_ambiguous_fix_carries_its_solutions_disagreement(p):
    """The fused solution's noise gains the squared gap between the two solutions."""
    ekf = _running(p)
    a, b = _fix(-0.5, 0.0, 0.0), _fix(-0.48, 0.0, 5.0)
    fix, _, _ = ekf_node.resolve(ekf, [(a, True), (b, True)], True, 10 / 150.0, True)
    gap = np.array([0.02, 0.0, math.radians(5.0)])
    assert np.allclose(fix[1], R + np.diag(gap * gap))
    single, _, _ = ekf_node.resolve(ekf, [(a, True)], False, 10 / 150.0, True)
    assert np.allclose(single[1], R)                    # one solution: its own noise only


def test_fixes_agree_compares_each_with_the_one_before():
    zs = [np.array([0.0, 0.01 * k, math.radians(0.5 * k)]) for k in range(5)]
    assert ekf_node.fixes_agree(zs, 0.03, math.radians(3.0))
    zs[2] = zs[2] + np.array([0.0, 0.05, 0.0])
    assert not ekf_node.fixes_agree(zs, 0.03, math.radians(3.0))


def _result_at(node, x, y, theta_deg, t):
    """Return an unambiguous result for the robot at (x, y, theta) in the target frame at t [s]."""
    from convchart_interfaces.msg import RosInferenceResult
    chain = node.chain
    T_T_board = chain.T_TM @ se3(R_UPRIGHT, -R_UPRIGHT @ chain.c_M)
    T_OM = chain.T_OB @ se3_inv(se3(rot_z(math.radians(theta_deg)), [x, y, 0.0])) @ T_T_board
    msg = RosInferenceResult(ambiguous=False, covariance_valid=True)
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
    _pose(T_OM, msg.pose)
    return msg


def test_the_filter_re_anchors_after_rejected_fixes_that_agree(node):
    """A 15 mm, 3 deg jump (a bump at the dock): five gated fixes that agree restart the filter."""
    node.active = node.zeroed = True
    node.on_pose(_result_at(node, -0.6, 0.0, 0.0, 100.0))
    assert node.ekf.initialised and abs(node.ekf.x[1]) < 1e-6
    for k in range(1, ekf_node.REANCHOR_AFTER):
        node.on_pose(_result_at(node, -0.6, 0.015, 3.0, 100.0 + k / 15.0))
        assert not node.last_accepted and abs(node.ekf.x[1]) < 0.002     # rejected, not moved
    node.on_pose(_result_at(node, -0.6, 0.015, 3.0, 100.0 + ekf_node.REANCHOR_AFTER / 15.0))
    assert node.reanchors == 1
    assert np.allclose(node.ekf.x[:3], (-0.6, 0.015, math.radians(3.0)), atol=1e-4)


def test_scattered_rejected_fixes_do_not_re_anchor(node):
    """Fixes that disagree with each other as much as with the filter are outliers: no restart."""
    node.active = node.zeroed = True
    node.on_pose(_result_at(node, -0.6, 0.0, 0.0, 100.0))
    for k in range(1, 2 * ekf_node.REANCHOR_AFTER):
        y = 0.04 if k % 2 else -0.04
        node.on_pose(_result_at(node, -0.6, y, 0.0, 100.0 + k / 15.0))
    assert node.reanchors == 0 and abs(node.ekf.x[1]) < 0.002


def test_while_acquiring_a_pose_is_fused_without_the_gate(node):
    """In ACQUIRE the filter may move to the other solution; once moving, the gate is back."""
    node.active = node.zeroed = True
    node.acquiring = True
    node.on_pose(_result_at(node, -0.6, 0.0, 0.0, 100.0))
    node.on_pose(_result_at(node, -0.6, 0.06, 6.0, 100.0 + 1 / 15.0))     # NIS far over the gate
    assert node.last_accepted and node.ekf.x[1] > 0.02
    node.acquiring = False
    y = node.ekf.x[1]
    node.on_pose(_result_at(node, -0.6, y - 0.06, -6.0, 100.0 + 2 / 15.0))
    assert not node.last_accepted and abs(node.ekf.x[1] - y) < 0.002
