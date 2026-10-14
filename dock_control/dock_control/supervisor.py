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

EST_TIMEOUT_S = 0.1     # an estimate older than this is stale: 15 IMU periods, 2 ticks
IMU_RATE_TOL = 0.1      # start moving only with the IMU no more than 10 % below imu.rate


class Stage(enum.IntEnum):
    """Supervisor stages; the values match the constants in dock_msgs/DockStage.msg."""

    IDLE = 0
    ACQUIRE = 1
    APPROACH = 2
    FINAL = 3
    DOCKED = 4
    HOLD = 5
    ABORT = 6


# The stages that command motion, and so the ones an unsafe estimate aborts.
MOVING = (Stage.APPROACH, Stage.FINAL, Stage.HOLD)


@dataclass
class Inputs:
    """Everything step() reads, kept current by mpc_node."""

    now: float = 0.0            # ROS time [s]
    armed: bool = False         # operator request, dock/arm
    drive_ok: bool = False      # serial bridge armed and streaming (STALLED or RUNNING)
    x: object = None            # latest estimate [x, y, theta, vx, vy]; None until the EKF starts
    p_diag: object = None       # its covariance diagonal
    est_age: float = math.inf   # seconds since that estimate's IMU sample
    imu_rate: float = 0.0       # IMU samples per second, as the EKF measures them
    fix_age: float = math.inf   # seconds since the last accepted fix
    roll: float = 0.0           # monitored only (brief Section 2)
    pitch: float = 0.0
    u_last: object = None       # last body command [u_b, v_b, r_b]
    mpc_misses: int = 0         # consecutive ticks with a fallback or a blown budget
    r_gyro: float = 0.0         # yaw rate [rad/s], for the settle check at the standoff


@dataclass
class Output:
    """What one DockStage message carries."""

    stage: Stage = Stage.IDLE
    y_ref: tuple = (0.0, 0.0, 0.0)
    v_max: float = 0.0
    r_max: float = 0.0
    command_enabled: bool = False
    qw: tuple = (1.0, 1.0, 1.0)     # tracking multipliers [x, y, theta] for this tick's MPC
    l1_cap: bool = False            # cap |u_b| + |v_b| (HOLD), not each axis
    v_lat_max: object = None        # FINAL: |v_b| capped apart from |u_b| (None: v_max)


class Supervisor:
    """The stage machine of brief Section 7."""

    def __init__(self, p):
        sup, lim, tol, mpc = p['sup'], p['lim'], p['tol'], p['mpc']

        # --- constant, built once ---
        self.imu_rate_min = (1.0 - IMU_RATE_TOL) * p['imu']['rate']
        self.standoff_ref = (-sup['standoff'], 0.0, 0.0)
        self.dock_ref = (0.0, 0.0, 0.0)
        self.align_tol_y = sup['align_tol_m']                 # APPROACH -> FINAL
        self.align_tol_th = math.radians(sup['align_tol_deg'])
        self.standoff_tol = sup['standoff_tol']
        self.dock_tol = sup['dock_tol']                       # FINAL -> DOCKED
        self.dock_vtol_sq = sup['dock_vtol'] ** 2
        self.tgt_y = tol['tgt_y']
        self.tgt_th = math.radians(tol['tgt_th_deg'])
        self.hold_after_s = sup['hold_after_s']
        self.abort_after_s = sup['abort_after_s']
        self.v_approach = lim['v_approach']
        self.v_final = lim['v_final']
        self.r_max = lim['r_max']
        # Tracking multipliers per stage (mpc.qw, mpc.qw_final): APPROACH may arc in
        # while it turns, FINAL locks the heading (the user's design, 2026-10-13).
        self.qw_approach = _triple(mpc.get('qw', 1.0))
        self.qw_final = _triple(mpc.get('qw_final', mpc.get('qw', 1.0)))
        # APPROACH's multipliers move from qw (far) to qw_near as the distance still to go to
        # the standoff, along x, falls from qw_ramp_m[0] to qw_ramp_m[1] (smoothstep): the
        # lateral weight stays low far out, so the plan does not strafe first, and rises in
        # time for the robot to reach the standoff on the centre line instead of still
        # strafing across it (the user, 2026-10-14: "it needs to correct in a bit earlier").
        # Without qw_ramp_m, APPROACH keeps qw.
        self.qw_near = _triple(mpc.get('qw_near', mpc.get('qw', 1.0)))
        ramp = sup.get('qw_ramp_m')
        self.qw_ramp = (float(ramp[0]), float(ramp[1])) if ramp else None
        # APPROACH's heading reference (sup.look_at_m = [free, lock], the distance still to go
        # to the standoff): far out, the bearing from the camera to the board centre, so the
        # camera keeps the board in the middle of its view while the robot arcs in on a
        # holonomic path; blending (smoothstep) to the board's normal, theta = 0, from free
        # in to lock. The user, 2026-10-14: the robot "didnt turn enough during arcing,
        # meaning it lost the board ... we want it to track the board". Without look_at_m,
        # APPROACH aims at theta = 0 throughout.
        look = sup.get('look_at_m')
        self.look_at_m = (float(look[0]), float(look[1])) if look else None
        g = p['geom']
        self.marker_xy = (float(g['marker_m'][0]), float(g['marker_m'][1]))
        self.cam_xy = (float(g['cam_xyz'][0]), float(g['cam_xyz'][1]))
        # The standoff hands over to FINAL only once the robot has settled there aligned:
        # speed under align_vtol, yaw rate under align_rtol_deg, for align_dwell_s. Then FINAL
        # drives straight in with |v_b| capped at lim.v_lat_final, so it does not strafe at
        # the coupler (the user, 2026-10-14: "an approach stage where we enforce angular and
        # lateral alignment before moving straight forward into the final dock point").
        self.align_vtol_sq = float(sup.get('align_vtol', math.inf)) ** 2
        self.align_rtol = math.radians(float(sup.get('align_rtol_deg', math.inf)))
        self.align_dwell = float(sup.get('align_dwell_s', 0.0))
        self.v_lat_final = lim.get('v_lat_final')
        # APPROACH's speed cap falls (smoothstep) from v_approach to lim.v_align as the distance
        # still to go to the standoff falls from sup.slow_m[0] to slow_m[1], never faster than
        # one tick's increments: the drive answers late and short, and at full speed the plan
        # overshoots the standoff sideways and hunts there (simulated, 2026-10-14).
        slow = sup.get('slow_m')
        self.slow_m = (float(slow[0]), float(slow[1])) if slow else None
        self.v_align = float(lim.get('v_align', lim['v_approach']))
        # translational increment du_max = T_s a_max, for the FINAL and HOLD cap ramp
        self.du_t = mpc['Ts'] * lim['a_max']
        # Abort when the position variance passes (0.05 m)^2 (brief Sections 7 and 12).
        self.p_abort = 0.05 ** 2

        # --- state ---
        self.out = Output()
        self.entered_at = 0.0       # when the current stage was entered
        self.resume = Stage.APPROACH    # the stage HOLD returns to
        self.hold_ref = self.dock_ref   # the pose HOLD keeps
        self.aligned_since = None       # when the robot last became settled and aligned

    def step(self, inp):
        """
        Advance the stage machine one tick and return the Output to publish.

        The transitions are the table of brief Section 7. Disarming returns to
        IDLE from any stage; DOCKED and ABORT stay until the operator disarms.
        """
        stage = self.out.stage
        if not inp.armed:
            nxt = Stage.IDLE
        elif stage in MOVING and self._unsafe(inp):
            nxt = Stage.ABORT
        elif stage == Stage.IDLE:
            nxt = Stage.ACQUIRE
        elif stage == Stage.ACQUIRE:
            # the EKF publishes nothing until its first accepted fix
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
        """
        Return whether to abort a moving stage: drive lost, estimate stale, or covariance.

        Drive lost, estimate stale, or position variance past (0.05 m)^2 abort at
        once. The heading is not a reason to abort: keeping it aligned with the
        board is the MPC's job, through the heading term of its cost. (A
        corridor abort here stopped every start more than 20 deg off the board's
        normal on the robot, 2026-10-13, since the robot could not turn back in
        before it fired.)
        """
        return (not inp.drive_ok or inp.est_age > EST_TIMEOUT_S
                or inp.p_diag[0] > self.p_abort or inp.p_diag[1] > self.p_abort)

    def _ready(self, inp):
        """Return whether to start moving: estimate fresh, IMU >= 90 % of its rate, drive armed."""
        return (inp.x is not None and inp.drive_ok and inp.est_age <= EST_TIMEOUT_S
                and inp.imu_rate >= self.imu_rate_min)

    def _aligned(self, x):
        """
        Return whether the robot is aligned at the standoff.

        |y| < 10 mm, |theta| < 1 deg and |x + 0.30| < 50 mm (brief Section 7).
        """
        return (abs(x[1]) < self.align_tol_y and abs(x[2]) < self.align_tol_th
                and abs(x[0] - self.standoff_ref[0]) < self.standoff_tol)

    def _settled_aligned(self, inp):
        """Return whether the robot has stood aligned at the standoff for align_dwell_s."""
        x = inp.x
        if not (self._aligned(x) and x[3] * x[3] + x[4] * x[4] < self.align_vtol_sq
                and abs(inp.r_gyro) < self.align_rtol):
            self.aligned_since = None
            return False
        if self.aligned_since is None:
            self.aligned_since = inp.now
        return inp.now - self.aligned_since >= self.align_dwell

    def _docked(self, x):
        """Return whether docked: x >= -3 mm, speed < 20 mm/s, |y| <= 4 mm, |theta| <= 1.5 deg."""
        return (x[0] >= -self.dock_tol and x[3] * x[3] + x[4] * x[4] < self.dock_vtol_sq
                and abs(x[1]) <= self.tgt_y and abs(x[2]) <= self.tgt_th)

    def _ramped_cap(self, u_last):
        """
        Return the falling speed cap of FINAL and HOLD, never stepped.

        FINAL and HOLD cap the sum |u_b| + |v_b| (Output.l1_cap; APPROACH caps
        each axis on its own). A cap below what one tick of increments can reach
        leaves the hard amplitude rows infeasible. Each of |u_b| and |v_b| can
        fall by at most du_max, so the cap falls by min(|u_b|, du_max) +
        min(|v_b|, du_max): the brief's 2 du_max when both are moving, but only
        du_max when driving straight in (v_b = 0), where 2 du_max cannot be
        reached. It starts from the sum APPROACH left, which per axis can be
        above v_approach.
        """
        if u_last is None:
            return self.v_final
        ub, vb = abs(u_last[0]), abs(u_last[1])
        return max(self.v_final, ub + vb - min(ub, self.du_t) - min(vb, self.du_t))

    def approach_qw(self, x):
        """Return APPROACH's tracking multipliers at estimate x (qw_ramp_m: from qw to qw_near)."""
        if self.qw_ramp is None or x is None:
            return self.qw_approach
        far, near = self.qw_ramp
        to_go = max(0.0, self.standoff_ref[0] - float(x[0]))
        s = min(1.0, max(0.0, (far - to_go) / (far - near)))
        s = s * s * (3.0 - 2.0 * s)
        return tuple(a + s * (b - a) for a, b in zip(self.qw_approach, self.qw_near))

    def look_at(self, x):
        """Return APPROACH's heading reference at estimate x (look_at_m: bearing to theta = 0)."""
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
        """Return FINAL's (v_max, v_lat_max): each axis falling by du_max a tick to its own cap."""
        ub, vb = (0.0, 0.0) if u_last is None else (abs(u_last[0]), abs(u_last[1]))
        return (max(self.v_final, min(self.v_approach, ub - self.du_t)),
                max(self.v_lat_final, vb - self.du_t))

    def approach_cap(self, x, u_last):
        """Return APPROACH's per-axis speed cap at estimate x, falling no faster than du_max."""
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
    """Return a per-axis weight as a tuple of three floats (a scalar applies to all three)."""
    a = np.broadcast_to(np.asarray(w, dtype=float), (3,))
    return tuple(float(v) for v in a)
