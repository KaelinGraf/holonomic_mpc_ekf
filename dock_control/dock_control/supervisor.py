"""
Docking supervisor (report Section 3.6, brief Section 7), without ROS.

A library, not a node: mpc_node calls step() at the start of each tick, so the
reference and caps the MPC solves with are never a tick old.

It does two different jobs:

* Modes the MPC cannot decide for itself: do not move until the operator has
  armed and the EKF has started (IDLE, ACQUIRE); stop commanding once docked,
  so the integral action does not push against the coupler (DOCKED); hold or
  abort when the estimate can no longer be trusted (HOLD, ABORT).
* The reference and cap schedule (APPROACH to FINAL): align at the standoff
  under the approach cap, then go straight in under the final cap, which keeps
  the contact speed, and so the unmodelled wheel lag's overshoot (about
  v_final tau), small. This part could be a continuous schedule, such as a cap
  falling with distance, without changing Output; the verified design and the
  report's results use the two stages.

It also guards the estimate itself: motion starts only with the IMU at 90 % of
its configured rate or faster (the EKF steps each sample's interval, so a fast
IMU is no reason to wait, and a slow one is the sign of a failing node), and any
moving stage aborts once the estimate stops arriving (a dead IMU node would
otherwise leave the MPC steering on a frozen state).

Thresholds are converted to the units the comparisons use once, in __init__.
"""

from dataclasses import dataclass
import enum
import math

import numpy as np

EST_TIMEOUT_S = 0.1
IMU_RATE_TOL = 0.1


class Stage(enum.IntEnum):
    """Supervisor stages; the values match the constants in dock_msgs/DockStage.msg."""

    IDLE = 0
    ACQUIRE = 1
    APPROACH = 2
    FINAL = 3
    DOCKED = 4
    HOLD = 5
    ABORT = 6


MOVING = (Stage.APPROACH, Stage.FINAL, Stage.HOLD)


@dataclass
class Inputs:
    """Everything step() reads, kept current by mpc_node.

    Args:
        now: ROS time [s].
        armed: Operator request, from dock/arm.
        drive_ok: Whether the serial bridge is armed and streaming (STALLED or RUNNING).
        x: Latest estimate [x, y, theta, vx, vy]; None until the EKF starts.
        p_diag: Covariance diagonal of that estimate.
        est_age: Time since that estimate's IMU sample [s].
        imu_rate: IMU samples per second, as the EKF measures them [Hz].
        fix_age: Time since the last accepted fix [s].
        roll: Roll of the latest fix [rad], monitored only.
        pitch: Pitch of the latest fix [rad], monitored only.
        u_last: Last body command [u_b, v_b, r_b].
        mpc_misses: Consecutive ticks with a fallback or a blown budget.
        r_gyro: Yaw rate [rad/s], for the settle check at the standoff.
    """

    now: float = 0.0
    armed: bool = False
    drive_ok: bool = False
    x: object = None
    p_diag: object = None
    est_age: float = math.inf
    imu_rate: float = 0.0
    fix_age: float = math.inf
    roll: float = 0.0
    pitch: float = 0.0
    u_last: object = None
    mpc_misses: int = 0
    r_gyro: float = 0.0


@dataclass
class Output:
    """What one DockStage message carries.

    Args:
        stage: Current stage.
        y_ref: Pose reference [x, y, theta] in the target frame [m, m, rad].
        v_max: Translational speed cap [m/s].
        r_max: Yaw-rate cap [rad/s].
        command_enabled: Whether the MPC commands the drive this tick.
        qw: Tracking multipliers [x, y, theta] for this tick's MPC.
        l1_cap: Whether to cap |u_b| + |v_b| rather than each axis on its own.
        v_lat_max: Cap on |v_b| apart from |u_b| in FINAL [m/s]; None uses v_max.
    """

    stage: Stage = Stage.IDLE
    y_ref: tuple = (0.0, 0.0, 0.0)
    v_max: float = 0.0
    r_max: float = 0.0
    command_enabled: bool = False
    qw: tuple = (1.0, 1.0, 1.0)
    l1_cap: bool = False
    v_lat_max: object = None


class Supervisor:
    """Docking stage machine: the stage, reference and speed caps for each MPC tick.

    Args:
        p: Parameters from params.load (the sup, lim, tol, mpc, imu and geom sections).
    """

    def __init__(self, p):
        sup, lim, tol, mpc = p['sup'], p['lim'], p['tol'], p['mpc']

        self.imu_rate_min = (1.0 - IMU_RATE_TOL) * p['imu']['rate']
        self.standoff_ref = (-sup['standoff'], 0.0, 0.0)
        self.dock_ref = (0.0, 0.0, 0.0)
        self.align_tol_y = sup['align_tol_m']
        self.align_tol_th = math.radians(sup['align_tol_deg'])
        self.standoff_tol = sup['standoff_tol']
        self.dock_tol = sup['dock_tol']
        self.dock_vtol_sq = sup['dock_vtol'] ** 2
        self.tgt_y = tol['tgt_y']
        self.tgt_th = math.radians(tol['tgt_th_deg'])
        self.hold_after_s = sup['hold_after_s']
        self.abort_after_s = sup['abort_after_s']
        self.v_approach = lim['v_approach']
        self.v_final = lim['v_final']
        self.r_max = lim['r_max']
        self.qw_approach = _triple(mpc.get('qw', 1.0))
        self.qw_final = _triple(mpc.get('qw_final', mpc.get('qw', 1.0)))
        self.qw_near = _triple(mpc.get('qw_near', mpc.get('qw', 1.0)))
        ramp = sup.get('qw_ramp_m')
        self.qw_ramp = (float(ramp[0]), float(ramp[1])) if ramp else None
        look = sup.get('look_at_m')
        self.look_at_m = (float(look[0]), float(look[1])) if look else None
        g = p['geom']
        self.marker_xy = (float(g['marker_m'][0]), float(g['marker_m'][1]))
        self.cam_xy = (float(g['cam_xyz'][0]), float(g['cam_xyz'][1]))
        self.align_vtol_sq = float(sup.get('align_vtol', math.inf)) ** 2
        self.align_rtol = math.radians(float(sup.get('align_rtol_deg', math.inf)))
        self.align_dwell = float(sup.get('align_dwell_s', 0.0))
        self.v_lat_final = lim.get('v_lat_final')
        slow = sup.get('slow_m')
        self.slow_m = (float(slow[0]), float(slow[1])) if slow else None
        self.v_align = float(lim.get('v_align', lim['v_approach']))
        self.du_t = mpc['Ts'] * lim['a_max']
        self.p_abort = 0.05 ** 2
        start = sup.get('start_sigma_m')
        self.p_start = float(start) ** 2 if start else math.inf

        self.out = Output()
        self.entered_at = 0.0
        self.resume = Stage.APPROACH
        self.hold_ref = self.dock_ref
        self.aligned_since = None

    def step(self, inp):
        """Advance the stage machine one tick and return the Output to publish.

        Disarming returns to IDLE from any stage; DOCKED and ABORT stay until the operator disarms.

        Args:
            inp: This tick's Inputs.
        """
        stage = self.out.stage
        if not inp.armed:
            nxt = Stage.IDLE
        elif stage in MOVING and self._unsafe(inp):
            nxt = Stage.ABORT
        elif stage == Stage.IDLE:
            nxt = Stage.ACQUIRE
        elif stage == Stage.ACQUIRE:
            nxt = Stage.APPROACH if self._ready(inp) else stage
        elif stage in (Stage.APPROACH, Stage.FINAL) and (
                inp.fix_age > self.hold_after_s or inp.mpc_misses >= 2):
            nxt = Stage.HOLD
        elif stage == Stage.APPROACH:
            nxt = Stage.FINAL if self._settled_aligned(inp) else stage
        elif stage == Stage.FINAL:
            nxt = Stage.DOCKED if self._docked(inp.x) else stage
        elif stage == Stage.HOLD:
            if inp.fix_age <= self.hold_after_s and inp.mpc_misses == 0:
                nxt = self.resume
            elif inp.now - self.entered_at > self.abort_after_s:
                nxt = Stage.ABORT
            else:
                nxt = stage
        else:
            nxt = stage

        if nxt != stage:
            self.entered_at = inp.now
            if nxt == Stage.HOLD:
                self.resume = stage
                self.hold_ref = (float(inp.x[0]), float(inp.x[1]), float(inp.x[2]))
        self.out = self._output(nxt, inp.u_last, inp.x)
        return self.out

    def _unsafe(self, inp):
        """Return whether to abort a moving stage: drive lost, estimate stale or too uncertain.

        Args:
            inp: This tick's Inputs.
        """
        return (not inp.drive_ok or inp.est_age > EST_TIMEOUT_S
                or inp.p_diag[0] > self.p_abort or inp.p_diag[1] > self.p_abort)

    def _ready(self, inp):
        """Return whether to start moving: estimate fresh and sure, IMU >= 90 %, drive armed.

        Args:
            inp: This tick's Inputs.
        """
        return (inp.x is not None and inp.drive_ok and inp.est_age <= EST_TIMEOUT_S
                and inp.imu_rate >= self.imu_rate_min
                and inp.p_diag[0] < self.p_start and inp.p_diag[1] < self.p_start)

    def _aligned(self, x):
        """Return whether the robot is aligned at the standoff.

        Tests |y| < align_tol_m, |theta| < align_tol_deg and |x - standoff| < standoff_tol.

        Args:
            x: Estimate [x, y, theta, vx, vy] in the target frame.
        """
        return (abs(x[1]) < self.align_tol_y and abs(x[2]) < self.align_tol_th
                and abs(x[0] - self.standoff_ref[0]) < self.standoff_tol)

    def _settled_aligned(self, inp):
        """Return whether the robot has stood aligned at the standoff for align_dwell_s.

        Args:
            inp: This tick's Inputs.
        """
        x = inp.x
        if not (self._aligned(x) and x[3] * x[3] + x[4] * x[4] < self.align_vtol_sq
                and abs(inp.r_gyro) < self.align_rtol):
            self.aligned_since = None
            return False
        if self.aligned_since is None:
            self.aligned_since = inp.now
        return inp.now - self.aligned_since >= self.align_dwell

    def _docked(self, x):
        """Return whether docked: x >= -dock_tol, speed < dock_vtol, |y|, |theta| in tol.tgt_*.

        Args:
            x: Estimate [x, y, theta, vx, vy] in the target frame.
        """
        return (x[0] >= -self.dock_tol and x[3] * x[3] + x[4] * x[4] < self.dock_vtol_sq
                and abs(x[1]) <= self.tgt_y and abs(x[2]) <= self.tgt_th)

    def _ramped_cap(self, u_last):
        """Return the falling speed cap of FINAL and HOLD, never stepped.

        The cap on |u_b| + |v_b| falls by min(|u_b|, du_max) + min(|v_b|, du_max) a tick.

        Args:
            u_last: Last body command [u_b, v_b, r_b]; None gives v_final.
        """
        if u_last is None:
            return self.v_final
        ub, vb = abs(u_last[0]), abs(u_last[1])
        return max(self.v_final, ub + vb - min(ub, self.du_t) - min(vb, self.du_t))

    def approach_qw(self, x):
        """Return APPROACH's tracking multipliers at estimate x (qw_ramp_m: from qw to qw_near).

        Args:
            x: Estimate [x, y, theta, vx, vy] in the target frame, or None.
        """
        if self.qw_ramp is None or x is None:
            return self.qw_approach
        far, near = self.qw_ramp
        to_go = max(0.0, self.standoff_ref[0] - float(x[0]))
        s = min(1.0, max(0.0, (far - to_go) / (far - near)))
        s = s * s * (3.0 - 2.0 * s)
        return tuple(a + s * (b - a) for a, b in zip(self.qw_approach, self.qw_near))

    def look_at(self, x):
        """Return APPROACH's heading reference at estimate x (look_at_m: bearing to theta = 0).

        Args:
            x: Estimate [x, y, theta, vx, vy] in the target frame, or None.
        """
        if self.look_at_m is None or x is None:
            return 0.0
        free, lock = self.look_at_m
        to_go = max(0.0, self.standoff_ref[0] - float(x[0]))
        w = min(1.0, max(0.0, (to_go - lock) / (free - lock)))
        w = w * w * (3.0 - 2.0 * w)
        c, s = math.cos(x[2]), math.sin(x[2])
        cam_x = x[0] + c * self.cam_xy[0] - s * self.cam_xy[1]
        cam_y = x[1] + s * self.cam_xy[0] + c * self.cam_xy[1]
        return w * math.atan2(self.marker_xy[1] - cam_y, self.marker_xy[0] - cam_x)

    def _straight_in_caps(self, u_last):
        """Return FINAL's (v_max, v_lat_max): each axis falling by du_max a tick to its own cap.

        Args:
            u_last: Last body command [u_b, v_b, r_b], or None.
        """
        ub, vb = (0.0, 0.0) if u_last is None else (abs(u_last[0]), abs(u_last[1]))
        return (max(self.v_final, min(self.v_approach, ub - self.du_t)),
                max(self.v_lat_final, vb - self.du_t))

    def approach_cap(self, x, u_last):
        """Return APPROACH's per-axis speed cap at estimate x, falling no faster than du_max.

        Args:
            x: Estimate [x, y, theta, vx, vy] in the target frame, or None.
            u_last: Last body command [u_b, v_b, r_b], or None.
        """
        if self.slow_m is None or x is None:
            return self.v_approach
        start, end = self.slow_m
        to_go = max(0.0, self.standoff_ref[0] - float(x[0]))
        s = min(1.0, max(0.0, (start - to_go) / (start - end)))
        s = s * s * (3.0 - 2.0 * s)
        cap = self.v_approach + s * (self.v_align - self.v_approach)
        fastest = 0.0 if u_last is None else max(abs(u_last[0]), abs(u_last[1]))
        return min(self.v_approach, max(cap, fastest - self.du_t))

    def _output(self, stage, u_last, x=None):
        if stage == Stage.APPROACH:
            ref = (self.standoff_ref[0], self.standoff_ref[1], self.look_at(x))
            return Output(stage, ref, self.approach_cap(x, u_last), self.r_max, True,
                          self.approach_qw(x))
        if stage == Stage.FINAL:
            if self.v_lat_final is not None:
                v_max, v_lat = self._straight_in_caps(u_last)
                return Output(stage, self.dock_ref, v_max, self.r_max, True, self.qw_final,
                              v_lat_max=v_lat)
            return Output(stage, self.dock_ref, self._ramped_cap(u_last), self.r_max, True,
                          self.qw_final, l1_cap=True)
        if stage == Stage.HOLD:
            qw = (self.qw_final if self.resume == Stage.FINAL
                  else self.approach_qw(self.hold_ref))
            return Output(stage, self.hold_ref, self._ramped_cap(u_last), self.r_max, True, qw,
                          l1_cap=True)
        return Output(stage)


def _triple(w):
    """Return a per-axis weight as a tuple of three floats (a scalar applies to all three).

    Args:
        w: Scalar or three per-axis values.
    """
    a = np.broadcast_to(np.asarray(w, dtype=float), (3,))
    return tuple(float(v) for v in a)
