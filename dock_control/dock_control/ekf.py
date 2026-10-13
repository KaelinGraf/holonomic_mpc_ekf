"""
Docking EKF (report Section 2, brief Section 5), without ROS.

State x = [x, y, theta, vx, vy] in the target frame; input u = [a_x, a_y, r],
the body-frame linear acceleration (gravity removed, start-up offset subtracted)
and the yaw rate. Each IMU sample advances the filter over its own interval
(advance), in Euler steps near the design's 5 ms, so the model is right at
whatever rate the IMU delivers. A fix is applied at its photo time and the
steps since replayed (update). While the robot is known to be still (its
commands have been zero for a moment), each step ends with the velocity reset
to zero (stopped), so a moving board shifts the position, not the velocity, and
a dropout cannot leave a stale velocity behind. predict() runs at the IMU rate, so everything
that does not change between steps is built once in __init__: the noise
matrices, the ring buffer, and F and G with their constant entries already set,
which leaves the step to write nine entries. H = [I3 0] is a selection, so
update() slices instead of multiplying: H x = x[:3], H P H' = P[:3, :3],
P H' = P[:, :3].
"""

import math

from dock_control.geometry import wrap_pi
import numpy as np


class RingBuffer:
    """
    The filter's history over the last buffer_s seconds, keyed by IMU stamp.

    Each slot holds a step's result (state and covariance) and what produced it
    (the input it held, its length, and whether it ended with the velocity
    reset), so that Ekf.update can roll back to a fix's photo time, correct
    there and replay the steps after it, resets included. Preallocated:
    push() overwrites the oldest slot, so nothing is allocated at the IMU rate. A
    fix is a few dozen steps old when it arrives, so scanning back from the
    newest entry is quicker than a binary search.
    """

    def __init__(self, depth, n=5, m=3):
        self.depth = depth
        self.t = np.full(depth, -math.inf)     # stamps [s]; -inf marks an empty slot
        self.x = np.zeros((depth, n))          # the state after the step
        self.P = np.zeros((depth, n, n))       # and its covariance
        self.u = np.zeros((depth, m))          # the input the step held
        self.h = np.zeros(depth)               # the step's length [s]; 0 where the filter started
        self.zv = np.zeros(depth, bool)        # the step ended with the velocity reset to zero
        self.head = -1                         # slot of the newest entry

    def push(self, stamp, x, P=None, u=None, h=0.0, zv=False):
        """Store a step at stamp in the oldest slot."""
        self.head = k = (self.head + 1) % self.depth
        self.t[k] = stamp
        self.x[k] = x
        if P is not None:
            self.P[k] = P
        if u is not None:
            self.u[k] = u
        self.h[k] = h
        self.zv[k] = zv

    def index_at(self, stamp):
        """Return the slot of the latest entry with t <= stamp; None before the buffer."""
        i = self.head
        for _ in range(self.depth):
            t = self.t[i]
            if t <= stamp:
                return None if t == -math.inf else i
            i = i - 1 if i else self.depth - 1
        return None

    def at(self, stamp):
        """
        Return the latest stored state with t <= stamp, or None if stamp predates the buffer.

        The row is a view into the buffer: use it before the next push().
        """
        i = self.index_at(stamp)
        return None if i is None else self.x[i]


class Ekf:
    """The filter: constants from the parameter file, then initialise, predict, update."""

    def __init__(self, p):
        imu, fid, ekf = p['imu'], p['fid'], p['ekf']

        # --- constant, built once ---
        # The MATLAB design steps a fixed 5 ms (its 200 Hz), and the floors below
        # are its per-step values from the NEES sweep. Here each sample's own
        # interval is stepped, in steps near step_s, and the floors are scaled by
        # h / step_s so the noise they add per second is the tuned one at any rate.
        self.step = ekf['step_s']
        self.buffer_s = ekf['buffer_s']
        var_a = imu['sigma_a'] ** 2
        var_r = math.radians(imu['sigma_r_deg']) ** 2
        # per-sample IMU noise inflated by q_scale, and the process-noise floor (report eq. 4)
        self.Sigma_u = imu['q_scale'] * np.diag([var_a, var_a, var_r])
        p_floor = ekf['p_floor']
        self.Q_floor = np.diag([p_floor, p_floor, ekf['q_th'], ekf['q_v'], ekf['q_v']])
        self.gate = fid['gate']
        self.sigma_v0_sq = ekf['sigma_v0'] ** 2
        # the velocity's standard deviation after a reset to zero (the robot known to be still)
        self.sigma_zv_sq = ekf.get('zupt_sigma_v', 0.005) ** 2
        # While it stands, the pose in the target frame changes only if the board moves (by
        # hand, at up to ~0.1 m/s): a random walk of that size lets the fixes carry it there.
        self.q_board = np.array([ekf.get('zupt_q_p', 0.05) ** 2, ekf.get('zupt_q_p', 0.05) ** 2,
                                 math.radians(ekf.get('zupt_q_th_deg', 10.0)) ** 2])
        self.I5 = np.eye(5)
        # F = I5 + h df/dx and G = h df/du for a step of h (report eq. 3). The
        # entries that are always 0 or 1 are set here; jacobians() writes the rest.
        self.F = np.eye(5)
        self.G = np.zeros((5, 3))

        # --- state ---
        self.x = np.zeros(5)
        self.P = np.zeros((5, 5))
        self.t = None               # the filter's clock: time of the last step [s]
        self.last_fix = -math.inf   # capture stamp of the newest fix applied [s]
        self.initialised = False
        # Room for buffer_s with samples at up to twice imu.rate: p4p_imu polls at
        # twice its rate (oversample 2), so it cannot publish faster.
        self.buffer = RingBuffer(round(2.0 * ekf['buffer_s'] * imu['rate']))

    def f(self, x, u, h):
        """Return the Euler step of report eq. 2 over h from (x, u), without touching the state."""
        c, s = math.cos(x[2]), math.sin(x[2])
        ax, ay, r = u
        return x + h * np.array((x[3], x[4], r, c * ax - s * ay, s * ax + c * ay))

    def jacobians(self, x, u, h):
        """
        Write the step-dependent entries of self.F and self.G at (x, u) for step h; return (F, G).

        Report eq. 3, evaluated at the estimate before the step.
        """
        c, s = math.cos(x[2]), math.sin(x[2])
        ax, ay = u[0], u[1]
        F, G = self.F, self.G
        F[0, 3] = F[1, 4] = h
        F[3, 2] = -h * (ax * s + ay * c)
        F[4, 2] = h * (ax * c - ay * s)
        G[2, 2] = h
        G[3, 0], G[3, 1] = h * c, -h * s
        G[4, 0], G[4, 1] = h * s, h * c
        return F, G

    def initialise(self, z, R, stamp):
        """
        Start the filter from the first accepted fix (report Section 2.5).

        x0 = [z; 0; 0], P0 = blkdiag(R, sigma_v0^2 I2), buffered at stamp so that
        the next fix finds an entry.
        """
        self.x[:3] = z
        self.x[3:] = 0.0
        self.P[:] = 0.0
        self.P[:3, :3] = R
        self.P[3, 3] = self.P[4, 4] = self.sigma_v0_sq
        self.buffer.push(stamp, self.x, self.P)
        self.t = stamp
        self.last_fix = stamp
        self.initialised = True

    def predict(self, u, stamp, stopped=False):
        """
        Propagate x and P to stamp in one step and buffer the result (report eq. 4).

        With h = stamp - t: x <- f(x, u, h) with the heading wrapped;
        P <- F P F' + G Sigma_u G' + (h / step_s) Q_floor, symmetrised, with F and G
        evaluated before the step. stopped: the robot is known to be still, so the
        step ends with the velocity reset to zero (zero_velocity).
        """
        h = stamp - self.t
        self._step(u, h)
        if stopped:
            self.zero_velocity(h)
        self.buffer.push(stamp, self.x, self.P, u, h, stopped)
        self.t = stamp

    def zero_velocity(self, h):
        """
        End a step of h with the robot still: velocity zero, and the board free to move.

        A reset, not a fix: the commands say the robot stands, so its velocity is
        known independently of everything else in the state (sigma zupt_sigma_v, no
        correlation). Its pose in the target frame can then change only if the board
        moves, so the pose takes a random walk (zupt_q_p, zupt_q_th_deg per root
        second) and the fixes carry it there, instead of turning into velocity.
        Applied at the end of each step while stopped, and again when a late fix
        replays that step.
        """
        self.x[3:] = 0.0
        self.P[3:, :] = 0.0
        self.P[:, 3:] = 0.0
        self.P[3, 3] = self.P[4, 4] = self.sigma_zv_sq
        self.P[0, 0] += self.q_board[0] * h
        self.P[1, 1] += self.q_board[1] * h
        self.P[2, 2] += self.q_board[2] * h

    def _step(self, u, h):
        """Propagate x and P over one step of h holding u, storing nothing."""
        F, G = self.jacobians(self.x, u, h)
        self.x = self.f(self.x, u, h)
        self.x[2] = wrap_pi(self.x[2])
        P = F @ self.P @ F.T + G @ self.Sigma_u @ G.T + (h / self.step) * self.Q_floor
        self.P = 0.5 * (P + P.T)

    def advance(self, u, stamp, stopped=False):
        """
        Predict over a sample's interval, holding u; return the steps taken.

        stopped: the robot is still over this interval (see predict).

        The filter's clock is the samples' stamps. Each interval is split into
        round(interval / step_s) equal steps, at least one, so no step is longer
        than 1.5 step_s and the buffer keeps an entry that often for the fixes to
        look up: one step a sample at the IMU's 150 Hz, more over a dropped
        sample, none for a repeated stamp. A gap longer than the buffer steps one
        buffer's worth and re-anchors the clock to the stamp.
        """
        if stamp <= self.t:
            return 0
        t0 = self.t
        gap = stamp - t0 > self.buffer_s
        end = t0 + self.buffer_s if gap else stamp
        steps = max(1, round((end - t0) / self.step))
        h = (end - t0) / steps
        for k in range(1, steps):
            self.predict(u, t0 + k * h, stopped)
        self.predict(u, end, stopped)
        if gap:
            self.t = stamp
        return steps

    def nis(self, z, R, stamp):
        """Return the NIS a fix would have, without applying it (inf before the buffer)."""
        innovation = self._innovation(z, R, stamp)
        if innovation is None:
            return math.inf
        nu, S = innovation
        return float(nu @ np.linalg.solve(S, nu))

    def update(self, z, R, stamp):
        """
        Correct with a fix captured at stamp; return (accepted, nis) (report eqs. 7-8).

        The fix describes the robot at its photo time, a few dozen steps ago over
        WiFi (140 ms median). So the filter rolls back to the step at the capture
        stamp and corrects there, with that step's own covariance: innovation with
        the heading wrapped, gate on nis = nu' S^-1 nu, Joseph-form covariance.
        Then it replays the stored steps after it, each with its own input and
        length, up to now. A late fix so leaves exactly the state an on-time one
        would. Rejected if the gate trips, the stamp predates the buffer, or the
        fix is older than one already applied: fixes arrive in order, and the
        replay from an older one would undo the newer correction.
        """
        b = self.buffer
        k = b.index_at(stamp)
        if k is None or stamp < self.last_fix:
            return False, math.nan
        nu, S = self._innovation_at(k, z, R)
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.gate:
            return False, nis
        K = np.linalg.solve(S, b.P[k, :3, :]).T     # P H' S^-1, without an inverse
        self.x = b.x[k] + K @ nu
        self.x[2] = wrap_pi(self.x[2])
        IKH = self.I5.copy()
        IKH[:, :3] -= K                             # I - K H
        P = IKH @ b.P[k] @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        self._replay_from(k)
        self.last_fix = stamp
        return True, nis

    def reanchor(self, z, R, stamp):
        """
        Restart the filter from a fix at its photo time, then replay the steps since.

        As initialise (pose z with covariance R, velocity zero with sigma_v0, no
        correlation), but placed at the buffered step of the capture stamp, so the
        IMU steps stored after it are replayed instead of skipped: a restart in the
        middle of a turn keeps the turn made since the photo. Initialises at stamp
        instead if the stamp predates the buffer.
        """
        k = self.buffer.index_at(stamp)
        if k is None:
            self.initialise(z, R, stamp)
            return
        self.x = np.concatenate((z, np.zeros(2)))
        self.x[2] = wrap_pi(self.x[2])
        self.P = np.zeros((5, 5))
        self.P[:3, :3] = R
        self.P[3, 3] = self.P[4, 4] = self.sigma_v0_sq
        self._replay_from(k)
        self.last_fix = stamp

    def _replay_from(self, k):
        """Store the current state in slot k, then replay the stored steps after it to now."""
        b = self.buffer
        b.x[k], b.P[k] = self.x, self.P
        while k != b.head:
            k = (k + 1) % b.depth
            self._step(b.u[k], b.h[k])
            if b.zv[k]:
                self.zero_velocity(b.h[k])
            b.x[k], b.P[k] = self.x, self.P

    def _innovation(self, z, R, stamp):
        k = self.buffer.index_at(stamp)
        return None if k is None else self._innovation_at(k, z, R)

    def _innovation_at(self, k, z, R):
        """Innovation and its covariance against buffer slot k, with that step's own P."""
        nu = z - self.buffer.x[k, :3]
        nu[2] = wrap_pi(nu[2])
        return nu, self.buffer.P[k, :3, :3] + R
