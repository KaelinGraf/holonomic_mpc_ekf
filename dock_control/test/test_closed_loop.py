"""
Closed loop without ROS (brief Section 11; mirrors test_mpc_closed_loop).

The EKF, supervisor and MPC dock a simulated demonstrator: pose kinematics with
first-order wheel loops (tau = 0.1 s, report Table 5), IMU samples with white
noise at 150 Hz, and fixes on the 15 Hz grid with the range-dependent noise and
the 50 ms delay of report Section 4.1. The EKF advances over each sample's
interval and the MPC ticks on its own 20 Hz grid with the newest estimate, as
ekf_node and mpc_node do. The pass criteria are those of report Table 4.
"""

import math

from dock_control.ekf import Ekf
from dock_control.fiducial import FiducialChain
from dock_control.supervisor import Inputs, Stage, Supervisor
import numpy as np
import pytest

TAU = 0.1
DELAY_S = 0.05


def test_docks_from_the_report_start_pose(p):
    """Dock from (-2 m, 0.3 m, 15 deg) with the estimator in the loop (report Table 4)."""
    r = _dock(p, (-2.0, 0.3, 15.0))
    assert r['stage'] == Stage.DOCKED
    assert abs(r['pose'][1]) < p['tol']['y']
    assert abs(r['pose'][2]) < math.radians(p['tol']['tgt_th_deg'])
    coast = p['lim']['v_final'] * TAU
    assert r['max_x'] <= p['mpc']['contact_band'] + coast
    assert (r['contact_speed'] or 0.0) < 1.5 * p['lim']['v_final']
    assert r['max_wheel'] <= p['lim']['w_max']
    assert r['fallbacks'] == 0 and r['rejected'] == 0


@pytest.mark.parametrize('start', [(-1.0, 0.3, 60.0), (-1.0, -0.3, -70.0)])
def test_docks_from_far_off_the_boards_normal(p, start):
    """No heading corridor: the cost alone brings a start 60-70 deg off the normal into line."""
    r = _dock(p, start, seconds=60.0)
    assert r['stage'] == Stage.DOCKED
    assert abs(r['pose'][1]) < p['tol']['y']
    assert abs(r['pose'][2]) < math.radians(p['tol']['tgt_th_deg'])
    assert r['max_x'] <= p['mpc']['contact_band'] + p['lim']['v_final'] * TAU
    assert r['max_wheel'] <= p['lim']['w_max']


@pytest.mark.parametrize('start', [(-0.55, -0.03, 1.0), (-1.0, 0.3, 40.0)])
def test_the_robots_lag_and_drift_neither_rock_the_heading_nor_leave_it_beside_the_dock(p, start):
    """The robot of 2026-10-13: wheels lagging 0.15 s, commands 50 ms late, drifting sideways."""
    r = _dock(p, start, seconds=60.0, tau=0.15, cmd_delay_s=0.05, drift=-0.15)
    assert r['stage'] == Stage.DOCKED
    assert r['final_theta_sd'] < math.radians(1.0)
    assert r['final_r_max'] < 0.5 * p['lim']['r_max']
    assert abs(r['y_in']) < 0.010


def _dock(p, start, seconds=40.0, tau=TAU, cmd_delay_s=0.0, drift=0.0):
    """Run the closed loop from start (x, y [m], theta [deg]); return where and how it ended."""
    pytest.importorskip('daqp')
    from dock_control.mpc import Mpc
    rng = np.random.default_rng(1)
    T = 1.0 / p['imu']['rate']
    delay = round(DELAY_S / T)
    sigma_a, sigma_r = p['imu']['sigma_a'], math.radians(p['imu']['sigma_r_deg'])
    marker = np.array(p['geom']['marker_m'])

    ekf, sup, mpc, chain = Ekf(p), Supervisor(p), Mpc(p), FiducialChain(p)
    pose = np.array([start[0], start[1], math.radians(start[2])])
    vel = np.zeros(3)
    cmd = np.zeros(3)
    sent = [np.zeros(3)] * (round(cmd_delay_s / T) + 1)
    final, y_in = [], None
    inp = Inputs(armed=True, drive_ok=True, u_last=cmd)
    pending, next_fix, next_tick, last_fix = [], 0.0, 0.0, -math.inf
    fallbacks = rejected = 0
    max_x, contact_speed, max_wheel = -math.inf, None, 0.0
    stage = Stage.IDLE

    for k in range(round(seconds / T)):
        t = k * T
        applied = sent[0] + np.array((0.0, drift * sent[0][0], 0.0))
        acc = (applied - vel) / tau
        a_body = (acc[0] - vel[2] * vel[1], acc[1] + vel[2] * vel[0])
        vel = vel + T * acc
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose = pose + T * np.array((vel[0] * c - vel[1] * s, vel[0] * s + vel[1] * c, vel[2]))
        if pose[0] >= 0.0 and contact_speed is None:
            contact_speed = math.hypot(vel[0], vel[1])
        if pose[0] >= -0.05 and y_in is None:
            y_in = pose[1]
        max_x = max(max_x, pose[0])

        if t >= next_fix:
            next_fix += 1.0 / p['fid']['rate']
            sp = chain.sigma_p(float(np.hypot(*(marker - pose[:2]))))
            z = pose + rng.normal(0.0, (sp, sp, chain.sigma_th))
            pending.append((k + delay, z, np.diag((sp * sp, sp * sp, chain.sigma_th ** 2)), t))

        u_imu = np.array((a_body[0] + rng.normal(0.0, sigma_a),
                          a_body[1] + rng.normal(0.0, sigma_a), vel[2] + rng.normal(0.0, sigma_r)))
        if ekf.initialised:
            ekf.advance(u_imu, t)
        while pending and pending[0][0] == k:
            _, z, R, t_capture = pending.pop(0)
            if not ekf.initialised:
                ekf.initialise(z, R, t_capture)
                last_fix = t_capture
            elif ekf.update(z, R, t_capture)[0]:
                last_fix = t_capture
            else:
                rejected += 1

        if t >= next_tick:
            next_tick += p['mpc']['Ts']
            inp.now = t
            if ekf.initialised:
                inp.x, inp.p_diag = ekf.x.copy(), ekf.P.diagonal().copy()
                inp.fix_age, inp.est_age, inp.imu_rate = t - last_fix, 0.0, 1.0 / T
            out = sup.step(inp)
            stage = out.stage
            if stage == Stage.DOCKED:
                break
            if out.command_enabled:
                res = mpc.step(ekf.x, u_imu[2], cmd, out.y_ref, out.v_max, out.r_max)
                fallbacks += res.fallback
                cmd[:] = res.u
            else:
                cmd[:] = 0.0
            max_wheel = max(max_wheel, float(np.abs(mpc.J_inv @ cmd).max()))
            if stage == Stage.FINAL:
                final.append((t, pose[2], cmd[2]))
        sent = sent[1:] + [cmd.copy()]

    settled = [(th, r) for tf, th, r in final if tf - final[0][0] >= 3.0] if final else []
    return {'stage': stage, 'pose': pose, 'max_x': max_x, 'contact_speed': contact_speed,
            'max_wheel': max_wheel, 'fallbacks': fallbacks, 'rejected': rejected, 't': t,
            'y_in': y_in,
            'final_theta_sd': float(np.std([th for th, _ in settled])) if settled else math.nan,
            'final_r_max': max((abs(r) for _, r in settled), default=math.nan)}
