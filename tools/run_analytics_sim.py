#!/usr/bin/env python3
"""
Worker for run_analytics.py: one commit's dock_control against one run, in a process of its own.

    python3 tools/run_analytics_sim.py replay <code dir> <request.json> <result.json>
    python3 tools/run_analytics_sim.py sim    <code dir> <request.json> <result.json>

<code dir> holds the dock_control package as it was at a commit (run_analytics.py writes it from git),
so a run is replayed and simulated with the code that ran, not today's. A process of its own because
each commit's package is imported as `dock_control`.

replay  Re-runs logged MPC ticks through that commit's Mpc: the estimate the tick used, the previous
        command, the stage's reference and caps. Returns how far each replayed plan (y_pred) and command
        lie from the logged ones. The commit, or the config variant, whose replay matches best is the one
        that ran (mpc_analysis_2026-10-14/replay_plans.py: a match is about 10 mm median).
sim     The closed loop of mpc_analysis_2026-10-14/sim.py from a start pose: that commit's Supervisor, Mpc,
        Ekf and FiducialChain on a plant with the run's own drive model. Per axis, the applied command is
        gain x command, held delay_cmd, then a first-order lag of time constant tau; a deadband; fixes at
        fix_rate with the chain's bench noise model, delivered fix_delay late, only while the board is in
        the camera's field of view; IMU noise from the config. No ambiguous fixes, no wheel slip.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np


def _import(code_dir: str):
    sys.path.insert(0, code_dir)
    from dock_control.ekf import Ekf
    from dock_control.fiducial import FiducialChain
    from dock_control.mpc import Mpc
    from dock_control.supervisor import Inputs, Stage, Supervisor
    return Ekf, FiducialChain, Mpc, Inputs, Stage, Supervisor


def _triple(w) -> np.ndarray:
    return np.broadcast_to(np.asarray(w, dtype=float), (3,)).copy()


def _mpc_step(mpc, x, r_gyro, u_prev, y_ref, v_max, r_max, qw):
    """Mpc.step with the per-stage tracking multipliers where that commit's Mpc takes them."""
    try:
        return mpc.step(x, r_gyro, u_prev, y_ref, v_max, r_max, qw)
    except TypeError:           # before 75feb05: step() has no qw; qw (if any) was fixed at start-up
        return mpc.step(x, r_gyro, u_prev, y_ref, v_max, r_max)


def replay(code_dir: str, req: dict) -> dict:
    """Re-run each logged tick; return per-tick plan and command differences."""
    _, _, Mpc, _, _, _ = _import(code_dir)
    p = req['config']
    mpc = Mpc(p)
    qw_app = _triple(p['mpc'].get('qw', 1.0))
    qw_fin = _triple(p['mpc'].get('qw_final', p['mpc'].get('qw', 1.0)))
    plan, cmd, ok = [], [], 0
    for tk in req['ticks']:
        qw = qw_fin if tk['qw_stage'] == 'FINAL' else qw_app
        res = _mpc_step(mpc, np.asarray(tk['x'], float), tk['r_gyro'], np.asarray(tk['u_prev'], float),
                        np.asarray(tk['y_ref'], float), tk['v_max'], tk['r_max'], qw)
        ylog = np.asarray(tk['y_pred'], float).reshape(-1, 3)
        if ylog.shape != res.y_pred.shape:
            plan.append(math.inf)
        else:
            plan.append(float(np.abs(res.y_pred[:, :2] - ylog[:, :2]).max()))
        cmd.append(float(np.abs(np.asarray(res.u) - np.asarray(tk['u'], float)).max()))
        ok += 1
    return {'plan_diff_m': plan, 'u_diff': cmd, 'replayed': ok}


def simulate(code_dir: str, req: dict) -> dict:
    """Closed loop from req['start'] on the run's drive model; log at log_dt (see the module doc)."""
    Ekf, FiducialChain, Mpc, Inputs, Stage, Supervisor = _import(code_dir)
    p = req['config']
    rng = np.random.default_rng(req.get('seed', 1))
    T = 1.0 / p['imu']['rate']
    fix_delay = round(req.get('fix_delay', 0.05) / T)
    dcmd = round(req.get('delay_cmd', 0.05) / T)
    gain = np.asarray(req.get('gain', (1.0, 1.0, 1.0)), float)
    tau = np.maximum(np.asarray(req.get('tau', (0.15, 0.15, 0.15)), float), T)
    db = np.asarray(req.get('db', (0.0, 0.0, 0.0)), float)
    fov = req.get('fov_deg')
    fix_rate = req.get('fix_rate', p['fid']['rate'])
    sigma_a, sigma_r = p['imu']['sigma_a'], math.radians(p['imu']['sigma_r_deg'])
    marker = np.array(p['geom']['marker_m'][:2], float)
    cam_x = p['geom']['cam_xyz'][0]
    ekf, sup, mpc, chain = Ekf(p), Supervisor(p), Mpc(p), FiducialChain(p)
    pose = np.array(req['start'], float)
    vel = np.zeros(3)
    cmd = np.zeros(3)
    hist = [np.zeros(3)] * (dcmd + 1)
    inp = Inputs(armed=True, drive_ok=True, u_last=cmd)
    pending, next_fix, next_tick, last_fix = [], 0.0, 0.0, -math.inf
    stage = Stage.IDLE
    stages, log = [], []
    unseen, max_bearing = 0, 0.0
    log_dt = req.get('log_dt', 0.05)
    seconds = req.get('seconds', 60.0)
    for k in range(round(seconds / T)):
        t = k * T
        applied = gain * hist[0]
        eff = np.where(np.abs(applied) < db, 0.0, applied)
        acc = (eff - vel) / tau
        a_body = (acc[0] - vel[2] * vel[1], acc[1] + vel[2] * vel[0])
        vel = vel + T * acc
        c, s = math.cos(pose[2]), math.sin(pose[2])
        pose = pose + T * np.array((vel[0] * c - vel[1] * s, vel[0] * s + vel[1] * c, vel[2]))
        lens = pose[:2] + cam_x * np.array((c, s))
        bearing = math.atan2(marker[1] - lens[1], marker[0] - lens[0]) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        max_bearing = max(max_bearing, abs(bearing))
        if t >= next_fix:
            next_fix += 1.0 / fix_rate
            if fov is None or abs(math.degrees(bearing)) <= fov:
                sp = chain.sigma_p(float(np.hypot(*(marker - pose[:2]))))
                z = pose + rng.normal(0.0, (sp, sp, chain.sigma_th))
                pending.append((k + fix_delay, z, np.diag((sp * sp, sp * sp, chain.sigma_th ** 2)), t))
            else:
                unseen += 1
        u_imu = np.array((a_body[0] + rng.normal(0.0, sigma_a), a_body[1] + rng.normal(0.0, sigma_a),
                          vel[2] + rng.normal(0.0, sigma_r)))
        if ekf.initialised:
            ekf.advance(u_imu, t)
        while pending and pending[0][0] <= k:
            _, z, R, t_capture = pending.pop(0)
            if not ekf.initialised:
                ekf.initialise(z, R, t_capture)
                last_fix = t_capture
            elif ekf.update(z, R, t_capture)[0]:
                last_fix = t_capture
        if t >= next_tick - 1e-9:
            next_tick += p['mpc']['Ts']
            inp.now = t
            if ekf.initialised:
                inp.x, inp.p_diag = ekf.x.copy(), ekf.P.diagonal().copy()
                inp.fix_age, inp.est_age, inp.imu_rate = t - last_fix, 0.0, 1.0 / T
            out = sup.step(inp)
            if out.stage != stage:
                stages.append((round(t, 3), out.stage.name))
            stage = out.stage
            if out.command_enabled:
                res = _mpc_step(mpc, ekf.x, u_imu[2], cmd, out.y_ref, out.v_max, out.r_max,
                                getattr(out, 'qw', None))
                cmd[:] = res.u
            else:
                cmd[:] = 0.0
            if not log or t - log[-1][0] >= log_dt - 1e-9:
                log.append((round(t, 4), *map(float, pose), *map(float, cmd), int(stage),
                            *map(float, ekf.x if ekf.initialised else np.full(5, np.nan))))
            if stage in (Stage.DOCKED, Stage.ABORT):
                break
        hist.append(cmd.copy())
        hist.pop(0)
    return {'stages': stages, 'final_stage': stage.name, 't_end': t, 'unseen': unseen,
            'max_bearing_deg': math.degrees(max_bearing), 'final_pose': [float(v) for v in pose],
            'log_columns': ['t', 'x', 'y', 'theta', 'u_b', 'v_b', 'r_b', 'stage',
                            'est_x', 'est_y', 'est_theta', 'est_vx', 'est_vy'],
            'log': log}


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] not in ('replay', 'sim'):
        sys.exit(__doc__)
    mode, code_dir, req_path, out_path = sys.argv[1:]
    req = json.loads(Path(req_path).read_text())
    out = replay(code_dir, req) if mode == 'replay' else simulate(code_dir, req)
    Path(out_path).write_text(json.dumps(out))


if __name__ == '__main__':
    main()
