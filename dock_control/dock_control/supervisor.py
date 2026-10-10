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

It also guards the estimate itself: motion starts only with the IMU at its
configured rate, and any moving stage aborts once the estimate stops arriving
(a dead IMU node would otherwise leave the MPC steering on a frozen state).

Thresholds are converted to the units the comparisons use once, in __init__.
"""

from dataclasses import dataclass
import enum
import math

EST_TIMEOUT_S = 0.1     # an estimate older than this is stale: 20 IMU periods, 2 ticks
IMU_RATE_TOL = 0.1      # start moving only with the IMU within 10 % of imu.rate


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


@dataclass
class Output:
    """What one DockStage message carries."""

    stage: Stage = Stage.IDLE
    y_ref: tuple = (0.0, 0.0, 0.0)
    v_max: float = 0.0
    r_max: float = 0.0
    command_enabled: bool = False


class Supervisor:
    """The stage machine of brief Section 7."""

    def __init__(self, p):
        sup, lim, tol, mpc = p['sup'], p['lim'], p['tol'], p['mpc']

        # --- constant, built once ---
        self.imu_rate = p['imu']['rate']
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
        self.th_max = math.radians(mpc['th_max_deg'])         # corridor, abort outside it
        # translational increment du_max = T_s a_max, for the FINAL and HOLD cap ramp
        self.du_t = mpc['Ts'] * lim['a_max']
        # Abort when the position variance passes (0.05 m)^2 (brief Sections 7 and 12).
        self.p_abort = 0.05 ** 2

        # --- state ---
        self.out = Output()
        self.entered_at = 0.0       # when the current stage was entered
        self.resume = Stage.APPROACH    # the stage HOLD returns to
        self.hold_ref = self.dock_ref   # the pose HOLD keeps
        self.off_corridor_since = None  # when the estimate left the heading corridor

    def step(self, inp):
        """
        Advance the stage machine one tick and return the Output to publish.

        The transitions are the table of brief Section 7. Disarming returns to
        IDLE from any stage; DOCKED and ABORT stay until the operator disarms.
        """
        stage = self.out.stage
        if inp.x is None or abs(inp.x[2]) <= self.th_max:
            self.off_corridor_since = None
        elif self.off_corridor_since is None:
            self.off_corridor_since = inp.now
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
            nxt = Stage.FINAL if self._aligned(inp.x) else stage
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
        self.out = self._output(nxt, inp.u_last)
        return self.out

    def _unsafe(self, inp):
        """
        Return whether to abort a moving stage: drive, estimate, covariance or corridor.

        Drive lost, estimate stale, or position variance past (0.05 m)^2 abort
        at once. The heading corridor is soft in the MPC, so a brief excursion past it is
        expected (a start near +-20 deg, the initial turn); only one that lasts
        longer than hold_after_s aborts.
        """
        off_corridor = (self.off_corridor_since is not None
                        and inp.now - self.off_corridor_since > self.hold_after_s)
        return (not inp.drive_ok or off_corridor or inp.est_age > EST_TIMEOUT_S
                or inp.p_diag[0] > self.p_abort or inp.p_diag[1] > self.p_abort)

    def _ready(self, inp):
        """Return whether motion may start: estimate fresh, IMU at its rate, drive armed."""
        return (inp.x is not None and inp.drive_ok and inp.est_age <= EST_TIMEOUT_S
                and abs(inp.imu_rate - self.imu_rate) <= IMU_RATE_TOL * self.imu_rate)

    def _aligned(self, x):
        """
        Return whether the robot is aligned at the standoff.

        |y| < 10 mm, |theta| < 1 deg and |x + 0.30| < 50 mm (brief Section 7).
        """
        return (abs(x[1]) < self.align_tol_y and abs(x[2]) < self.align_tol_th
                and abs(x[0] - self.standoff_ref[0]) < self.standoff_tol)

    def _docked(self, x):
        """Return whether docked: x >= -3 mm, speed < 20 mm/s, |y| <= 4 mm, |theta| <= 1.5 deg."""
        return (x[0] >= -self.dock_tol and x[3] * x[3] + x[4] * x[4] < self.dock_vtol_sq
                and abs(x[1]) <= self.tgt_y and abs(x[2]) <= self.tgt_th)

    def _ramped_cap(self, u_last):
        """
        Return the falling speed cap of FINAL and HOLD, never stepped.

        A cap below what one tick of increments can reach leaves the hard
        amplitude rows infeasible. Each of |u_b| and |v_b| can fall by at most
        du_max, so the cap falls by min(|u_b|, du_max) + min(|v_b|, du_max): the
        brief's 2 du_max when both are moving, but only du_max when driving
        straight in (v_b = 0), where 2 du_max cannot be reached.
        """
        if u_last is None:
            return self.v_final
        ub, vb = abs(u_last[0]), abs(u_last[1])
        reachable = ub + vb - min(ub, self.du_t) - min(vb, self.du_t)
        return max(self.v_final, min(self.v_approach, reachable))

    def _output(self, stage, u_last):
        if stage == Stage.APPROACH:
            return Output(stage, self.standoff_ref, self.v_approach, self.r_max, True)
        if stage == Stage.FINAL:
            return Output(stage, self.dock_ref, self._ramped_cap(u_last), self.r_max, True)
        if stage == Stage.HOLD:
            return Output(stage, self.hold_ref, self._ramped_cap(u_last), self.r_max, True)
        return Output(stage)
