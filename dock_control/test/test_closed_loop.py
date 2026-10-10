"""
Closed loop without ROS (brief Section 11; mirrors test_mpc_closed_loop).

The EKF, supervisor and MPC dock a simulated demonstrator: pose kinematics with
first-order wheel loops (tau = 0.1 s, report Table 5), IMU samples with white
noise at 200 Hz, and fixes on the 15 Hz grid with the range-dependent noise and
the 50 ms delay of report Section 4.1. The pass criteria are those of report
Table 4.
"""

import math

from dock_control.ekf import Ekf
from dock_control.fiducial import FiducialChain
from dock_control.supervisor import Inputs, Stage, Supervisor
import numpy as np
import pytest

TAU = 0.1               # wheel-loop time constant of the demonstrator [s]
DELAY_S = 0.05          # capture to availability [s]


def test_docks_from_the_report_start_pose(p):
    """Dock from (-2 m, 0.3 m, 15 deg) with the estimator in the loop (report Table 4)."""
    pytest.importorskip('daqp')
    from dock_control.mpc import Mpc
    rng = np.random.default_rng(1)
    T = 1.0 / p['imu']['rate']
    tick_every = round(p['mpc']['Ts'] / T)
    delay = round(DELAY_S / T)
    sigma_a, sigma_r = p['imu']['sigma_a'], math.radians(p['imu']['sigma_r_deg'])
    marker = np.array(p['geom']['marker_m'])
    v_final, x_max = p['lim']['v_final'], p['mpc']['contact_band']

    ekf, sup, mpc, chain = Ekf(p), Supervisor(p), Mpc(p), FiducialChain(p)
    pose = np.array([-2.0, 0.3, math.radians(15.0)])
    vel = np.zeros(3)                       # body velocity [u, v, r], lagging the command
    cmd = np.zeros(3)
    inp = Inputs(armed=True, drive_ok=True, u_last=cmd)
    pending, next_fix, last_fix = [], 0.0, -math.inf
    fallbacks = rejected = 0
    max_x, contact_speed, max_wheel = -math.inf, None, 0.0
    stage = Stage.IDLE

    for k in range(round(40.0 / T)):
        t = k * T
        # plant: wheel loops, then the pose kinematics
        acc = (cmd - vel) / TAU
        a_body = (acc[0] - vel[2] * vel[1], acc[1] + vel[2] * vel[0])
        vel = vel + T * acc
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose = pose + T * np.array((vel[0] * c - vel[1] * s, vel[0] * s + vel[1] * c, vel[2]))
        if pose[0] >= 0.0 and contact_speed is None:
            contact_speed = math.hypot(vel[0], vel[1])
        max_x = max(max_x, pose[0])

        # camera on the 15 Hz grid, delivered after the delay
        if t >= next_fix:
            next_fix += 1.0 / p['fid']['rate']
            sp = chain.sigma_p(float(np.hypot(*(marker - pose[:2]))))
            z = pose + rng.normal(0.0, (sp, sp, chain.sigma_th))
            pending.append((k + delay, z, np.diag((sp * sp, sp * sp, chain.sigma_th ** 2)), t))

        u_imu = np.array((a_body[0] + rng.normal(0.0, sigma_a),
                          a_body[1] + rng.normal(0.0, sigma_a), vel[2] + rng.normal(0.0, sigma_r)))
        if ekf.initialised:
            ekf.predict(u_imu, t)
        while pending and pending[0][0] == k:
            _, z, R, t_capture = pending.pop(0)
            if not ekf.initialised:
                ekf.initialise(z, R, t_capture)
                last_fix = t_capture
            elif ekf.update(z, R, t_capture)[0]:
                last_fix = t_capture
            else:
                rejected += 1

        if k % tick_every == 0:
            inp.now = t
            if ekf.initialised:
                inp.x, inp.p_diag = ekf.x.copy(), ekf.P.diagonal().copy()
                inp.fix_age = t - last_fix
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

    assert stage == Stage.DOCKED                                   # within 40 s
    # The lateral error is checked against the coupler tolerance: the 4 mm internal
    # target is estimator-limited on the demonstrator (26 of 30 runs, report 4.4).
    assert abs(pose[1]) < p['tol']['y']
    assert abs(pose[2]) < math.radians(p['tol']['tgt_th_deg'])
    assert max_x <= x_max + v_final * TAU                           # band plus the lag coast
    assert (contact_speed or 0.0) < 1.5 * v_final                  # none if it never touched
    assert max_wheel <= p['lim']['w_max']
    assert fallbacks == 0 and rejected == 0
