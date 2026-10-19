"""The supervisor against the stage table of brief Section 7."""

import copy
import math

from dock_control.supervisor import Inputs, Stage, Supervisor
import numpy as np
import pytest


def _inputs(**changes):
    """Return armed, healthy inputs with the robot 1 m out, changed as given.

    Args:
        **changes: Inputs fields to override.
    """
    values = {'now': 0.0, 'armed': True, 'drive_ok': True,
              'x': np.array([-1.0, 0.1, 0.05, 0.0, 0.0]), 'p_diag': np.full(5, 1e-6),
              'est_age': 0.004, 'imu_rate': 150.0, 'fix_age': 0.05, 'u_last': np.zeros(3)}
    values.update(changes)
    return Inputs(**values)


def _approaching(p):
    sup = Supervisor(p)
    sup.step(_inputs(x=None))
    assert sup.step(_inputs()).stage == Stage.APPROACH
    return sup


def test_stage_values_match_the_message():
    """Check that Stage and dock_msgs/DockStage agree on every value."""
    msg = pytest.importorskip('dock_msgs.msg')
    for stage in Stage:
        assert getattr(msg.DockStage, stage.name) == stage.value


def test_disarmed_means_idle_with_the_command_disabled(p):
    """Check that nothing moves until the operator arms, even with the drive ready."""
    out = Supervisor(p).step(Inputs(drive_ok=True))
    assert out.stage == Stage.IDLE and not out.command_enabled


def test_nominal_run_through_the_table(p):
    """Step through IDLE, ACQUIRE, APPROACH, FINAL, DOCKED, and back to IDLE on disarm."""
    for key in ('look_at_m', 'slow_m', 'align_vtol', 'align_rtol_deg', 'align_dwell_s'):
        p['sup'].pop(key, None)
    p['lim'].pop('v_lat_final', None)
    sup = Supervisor(p)
    assert sup.step(_inputs(x=None)).stage == Stage.ACQUIRE
    assert sup.step(_inputs(x=None)).stage == Stage.ACQUIRE
    out = sup.step(_inputs())
    assert out.stage == Stage.APPROACH and out.command_enabled
    s0 = p['sup']['standoff']
    assert out.y_ref == (-s0, 0.0, 0.0) and out.v_max == p['lim']['v_approach']
    assert not out.l1_cap

    va, du = p['lim']['v_approach'], p['mpc']['Ts'] * p['lim']['a_max']
    aligned = np.array([-s0 + 0.5 * p['sup']['standoff_tol'], 0.009, math.radians(0.9), 0.0, 0.0])
    out = sup.step(_inputs(x=aligned, u_last=np.array([va, 0.0, 0.0])))
    assert out.stage == Stage.FINAL and out.y_ref == (0.0, 0.0, 0.0)
    assert out.v_max == pytest.approx(va - du)
    assert out.l1_cap
    out = sup.step(_inputs(x=aligned, u_last=np.array([va - 0.06, 0.06, 0.0])))
    assert out.v_max == pytest.approx(va - 2 * du)
    out = sup.step(_inputs(x=aligned, u_last=np.array([va, 0.12, 0.0])))
    assert out.v_max == pytest.approx(va + 0.12 - 2 * du)
    out = sup.step(_inputs(x=aligned, u_last=np.array([0.02, 0.01, 0.0])))
    assert out.v_max == p['lim']['v_final']

    docked = np.array([-0.002, 0.003, math.radians(1.0), 0.01, 0.0])
    out = sup.step(_inputs(x=docked))
    assert out.stage == Stage.DOCKED and not out.command_enabled
    assert sup.step(_inputs(x=docked)).stage == Stage.DOCKED
    assert sup.step(_inputs(armed=False)).stage == Stage.IDLE


def test_hold_on_marker_loss_then_resume_or_abort(p):
    """Check that a lost fix holds the pose, then resumes within 2 s or else aborts."""
    sup = _approaching(p)
    out = sup.step(_inputs(now=1.0, fix_age=0.6))
    assert out.stage == Stage.HOLD and out.command_enabled
    assert out.y_ref == pytest.approx((-1.0, 0.1, 0.05))
    assert sup.step(_inputs(now=1.5)).stage == Stage.APPROACH
    assert sup.step(_inputs(now=2.0, fix_age=0.6)).stage == Stage.HOLD
    assert sup.step(_inputs(now=3.9, fix_age=2.5)).stage == Stage.HOLD
    assert sup.step(_inputs(now=4.1, fix_age=2.7)).stage == Stage.ABORT
    assert sup.step(_inputs(now=4.2)).stage == Stage.ABORT


def test_two_mpc_misses_hold(p):
    """Check that two consecutive fallbacks or blown budgets hold the robot."""
    sup = _approaching(p)
    assert sup.step(_inputs(mpc_misses=1)).stage == Stage.APPROACH
    assert sup.step(_inputs(mpc_misses=2)).stage == Stage.HOLD


@pytest.mark.parametrize('bad', [{'drive_ok': False}, {'p_diag': np.full(5, 0.06 ** 2)},
                                 {'est_age': 0.15}])
def test_abort_from_a_moving_stage(p, bad):
    """Check that a lost drive, a stale estimate or a variance over (0.05 m)^2 aborts at once."""
    assert _approaching(p).step(_inputs(**bad)).stage == Stage.ABORT


@pytest.mark.parametrize('bad', [{'imu_rate': 50.0}, {'imu_rate': 130.0}, {'imu_rate': 0.0},
                                 {'est_age': 0.15}, {'drive_ok': False}])
def test_motion_waits_for_a_healthy_imu_and_drive(p, bad):
    """Check that ACQUIRE waits for the IMU at 90 % of its rate, a fresh estimate and the drive."""
    sup = Supervisor(p)
    sup.step(_inputs(x=None))
    assert sup.step(_inputs(**bad)).stage == Stage.ACQUIRE
    assert sup.step(_inputs(imu_rate=140.0)).stage == Stage.APPROACH


def test_a_fast_imu_does_not_hold_the_start(p):
    """Check that an IMU faster than imu.rate does not hold the start."""
    sup = Supervisor(p)
    sup.step(_inputs(x=None))
    assert sup.step(_inputs(imu_rate=181.0)).stage == Stage.APPROACH


def test_the_heading_never_aborts_the_approach(p):
    """Check that a heading far off the board's normal never aborts APPROACH: the MPC aligns."""
    off = np.array([-0.45, -0.25, math.radians(32.0), 0.0, 0.0])
    sup = _approaching(p)
    for k in range(40):
        assert sup.step(_inputs(now=1.0 + 0.1 * k, x=off)).stage == Stage.APPROACH


def test_approach_weights_ramp_from_far_to_near_with_the_distance_to_go(p):
    """Check that APPROACH's multipliers ramp from mpc.qw to mpc.qw_near across sup.qw_ramp_m."""
    q = copy.deepcopy(p)
    q['mpc']['qw'], q['mpc']['qw_near'] = [1.0, 0.25, 2.0], [1.0, 1.0, 2.0]
    q['sup']['qw_ramp_m'] = [0.8, 0.25]
    sup = Supervisor(q)
    s0 = q['sup']['standoff']

    def at(to_go):
        return sup.approach_qw(np.array([-s0 - to_go, 0.05, 0.0, 0.0, 0.0]))
    assert at(1.5) == pytest.approx((1.0, 0.25, 2.0))
    assert at(0.8) == pytest.approx((1.0, 0.25, 2.0))
    assert at(0.25) == pytest.approx((1.0, 1.0, 2.0)) and at(0.0) == pytest.approx((1.0, 1.0, 2.0))
    assert at(0.525)[1] == pytest.approx(0.625)
    assert at(0.6)[1] < at(0.5)[1] < at(0.4)[1]
    sup.step(_inputs(x=None))
    out = sup.step(_inputs(x=np.array([-s0 - 0.3, 0.05, 0.0, 0.0, 0.0])))
    assert out.stage == Stage.APPROACH and out.qw == pytest.approx(at(0.3))
    q['sup'].pop('qw_ramp_m')
    plain = Supervisor(q)
    assert plain.approach_qw(np.array([-s0 - 0.3, 0.0, 0.0, 0.0, 0.0])) == plain.qw_approach


def _g(p):
    """Return a copy of p with the align-then-plunge settings.

    Args:
        p: Demo parameters.
    """
    q = copy.deepcopy(p)
    q['sup'].update(standoff=0.08, align_tol_m=0.010, align_tol_deg=2.0, standoff_tol=0.015,
                    align_vtol=0.03, align_rtol_deg=4.0, align_dwell_s=0.2,
                    look_at_m=[0.30, 0.06], slow_m=[0.30, 0.08])
    q['lim'].update(v_align=0.04, v_lat_final=0.02)
    return q


def test_approach_looks_at_the_board_far_out_and_straight_ahead_near_the_standoff(p):
    """Check that the heading reference is the bearing to the board far out, and zero near."""
    sup = Supervisor(_g(p))
    mx, my = p['geom']['marker_m']
    far = np.array([-1.0, 0.4, 0.0, 0.0, 0.0])
    cam_x = far[0] + p['geom']['cam_xyz'][0]
    assert sup.look_at(far) == pytest.approx(math.atan2(my - far[1], mx - cam_x))
    assert sup.look_at(np.array([-0.10, 0.02, 0.0, 0.0, 0.0])) == 0.0
    mid = sup.look_at(np.array([-0.08 - 0.18, 0.05, 0.0, 0.0, 0.0]))
    full = math.atan2(my - 0.05, mx - (-0.26 + p['geom']['cam_xyz'][0]))
    assert 0.0 < abs(mid) < abs(full)


def test_the_standoff_hands_over_only_once_settled_and_aligned(p):
    """Check that APPROACH hands over to FINAL only after align_dwell_s settled and aligned."""
    sup = Supervisor(_g(p))
    sup.step(_inputs(x=None))
    at = np.array([-0.08, 0.004, math.radians(0.5), 0.0, 0.0])
    out = sup.step(_inputs(x=at, now=1.0))
    assert out.stage == Stage.APPROACH
    moving = at.copy()
    moving[4] = 0.05
    assert sup.step(_inputs(x=moving, now=1.1)).stage == Stage.APPROACH
    assert sup.step(_inputs(x=at, now=1.2)).stage == Stage.APPROACH
    assert sup.step(_inputs(x=at, now=1.35)).stage == Stage.APPROACH
    out = sup.step(_inputs(x=at, now=1.45))
    assert out.stage == Stage.FINAL and not out.l1_cap
    assert out.v_lat_max == pytest.approx(0.02) and out.v_max == pytest.approx(0.09)


def test_approach_slows_near_the_standoff_no_faster_than_the_drive_can_brake(p):
    sup = Supervisor(_g(p))
    far = np.array([-0.08 - 0.5, 0.0, 0.0, 0.0, 0.0])
    near = np.array([-0.08 - 0.05, 0.0, 0.0, 0.0, 0.0])
    assert sup.approach_cap(far, np.zeros(3)) == pytest.approx(0.18)
    assert sup.approach_cap(near, np.zeros(3)) == pytest.approx(0.04)
    du = p['mpc']['Ts'] * p['lim']['a_max']
    assert sup.approach_cap(near, np.array([0.18, 0.0, 0.0])) == pytest.approx(0.18 - du)


def test_acquire_waits_until_the_estimate_is_sure_enough_to_move(p):
    """Check that ACQUIRE holds while the position sd is over sup.start_sigma_m."""
    q = copy.deepcopy(p)
    q['sup']['start_sigma_m'] = 0.03
    sup = Supervisor(q)
    sup.step(_inputs(x=None))
    out = sup.step(_inputs(p_diag=np.array([0.0002, 0.005, 1e-4, 1e-3, 1e-3])))
    assert out.stage == Stage.ACQUIRE and not out.command_enabled
    out = sup.step(_inputs(p_diag=np.full(5, 1e-6)))
    assert out.stage == Stage.APPROACH
