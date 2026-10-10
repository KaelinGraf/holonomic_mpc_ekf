"""The supervisor against the stage table of brief Section 7."""

import math

from dock_control.supervisor import Inputs, Stage, Supervisor
import numpy as np
import pytest


def _inputs(**changes):
    """Return armed, healthy inputs with the robot 1 m out, changed as given."""
    values = {'now': 0.0, 'armed': True, 'drive_ok': True,
              'x': np.array([-1.0, 0.1, 0.05, 0.0, 0.0]), 'p_diag': np.full(5, 1e-6),
              'fix_age': 0.05, 'u_last': np.zeros(3)}
    values.update(changes)
    return Inputs(**values)


def _approaching(p):
    sup = Supervisor(p)
    sup.step(_inputs(x=None))                   # IDLE -> ACQUIRE
    assert sup.step(_inputs()).stage == Stage.APPROACH
    return sup


def test_stage_values_match_the_message():
    """Stage and dock_msgs/DockStage agree on every value."""
    msg = pytest.importorskip('dock_msgs.msg')
    for stage in Stage:
        assert getattr(msg.DockStage, stage.name) == stage.value


def test_disarmed_means_idle_with_the_command_disabled(p):
    """Nothing moves until the operator arms, even with the drive ready."""
    out = Supervisor(p).step(Inputs(drive_ok=True))
    assert out.stage == Stage.IDLE and not out.command_enabled


def test_nominal_run_through_the_table(p):
    """IDLE, ACQUIRE, APPROACH, FINAL, DOCKED, and back to IDLE on disarm."""
    sup = Supervisor(p)
    assert sup.step(_inputs(x=None)).stage == Stage.ACQUIRE
    assert sup.step(_inputs(x=None)).stage == Stage.ACQUIRE     # waiting for the EKF
    out = sup.step(_inputs())
    assert out.stage == Stage.APPROACH and out.command_enabled
    assert out.y_ref == (-0.3, 0.0, 0.0) and out.v_max == p['lim']['v_approach']

    aligned = np.array([-0.26, 0.009, math.radians(0.9), 0.0, 0.0])
    out = sup.step(_inputs(x=aligned, u_last=np.array([0.3, 0.0, 0.0])))
    assert out.stage == Stage.FINAL and out.y_ref == (0.0, 0.0, 0.0)
    assert out.v_max == pytest.approx(0.275)        # one increment: v_b has none to give
    out = sup.step(_inputs(x=aligned, u_last=np.array([0.2, 0.1, 0.0])))
    assert out.v_max == pytest.approx(0.25)         # both axes moving: 2 du_max
    out = sup.step(_inputs(x=aligned, u_last=np.array([0.05, 0.02, 0.0])))
    assert out.v_max == p['lim']['v_final']         # floored at v_final

    docked = np.array([-0.002, 0.003, math.radians(1.0), 0.01, 0.0])
    out = sup.step(_inputs(x=docked))
    assert out.stage == Stage.DOCKED and not out.command_enabled
    assert sup.step(_inputs(x=docked)).stage == Stage.DOCKED          # until the operator
    assert sup.step(_inputs(armed=False)).stage == Stage.IDLE


def test_hold_on_marker_loss_then_resume_or_abort(p):
    """A lost fix holds the current pose; back within 2 s it resumes, otherwise it aborts."""
    sup = _approaching(p)
    out = sup.step(_inputs(now=1.0, fix_age=0.6))
    assert out.stage == Stage.HOLD and out.command_enabled
    assert out.y_ref == pytest.approx((-1.0, 0.1, 0.05))
    assert sup.step(_inputs(now=1.5)).stage == Stage.APPROACH
    assert sup.step(_inputs(now=2.0, fix_age=0.6)).stage == Stage.HOLD
    assert sup.step(_inputs(now=3.9, fix_age=2.5)).stage == Stage.HOLD
    assert sup.step(_inputs(now=4.1, fix_age=2.7)).stage == Stage.ABORT
    assert sup.step(_inputs(now=4.2)).stage == Stage.ABORT          # until the operator


def test_two_mpc_misses_hold(p):
    """Two consecutive fallbacks or blown budgets hold the robot."""
    sup = _approaching(p)
    assert sup.step(_inputs(mpc_misses=1)).stage == Stage.APPROACH
    assert sup.step(_inputs(mpc_misses=2)).stage == Stage.HOLD


@pytest.mark.parametrize('bad', [{'drive_ok': False}, {'p_diag': np.full(5, 0.06 ** 2)}])
def test_abort_from_a_moving_stage(p, bad):
    """Drive lost, or the position covariance above (0.05 m)^2, aborts at once."""
    assert _approaching(p).step(_inputs(**bad)).stage == Stage.ABORT


def test_abort_only_on_a_lasting_corridor_excursion(p):
    """Outside the heading corridor for longer than hold_after_s aborts; a brief excursion not."""
    off = np.array([-1.0, 0.1, math.radians(21.0), 0.0, 0.0])
    sup = _approaching(p)
    assert sup.step(_inputs(now=1.0, x=off)).stage == Stage.APPROACH
    assert sup.step(_inputs(now=1.3, x=off)).stage == Stage.APPROACH
    assert sup.step(_inputs(now=1.4)).stage == Stage.APPROACH       # back inside: reset
    assert sup.step(_inputs(now=1.5, x=off)).stage == Stage.APPROACH
    assert sup.step(_inputs(now=2.1, x=off)).stage == Stage.ABORT
