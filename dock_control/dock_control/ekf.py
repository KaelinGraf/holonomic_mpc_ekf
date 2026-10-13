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
    """The filter's history over the last buffer_s seconds, keyed by IMU stamp."""

    def __init__(self, depth, n=5, m=3):
        self.depth = depth
        self.t = np.full(depth, -math.inf)
        self.x = np.zeros((depth, n))
        self.P = np.zeros((depth, n, n))
        self.u = np.zeros((depth, m))
        self.h = np.zeros(depth)
        self.zv = np.zeros(depth, bool)
        self.head = -1

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

        self.step = ekf['step_s']
        self.buffer_s = ekf['buffer_s']
        var_a = imu['sigma_a'] ** 2
        var_r = math.radians(imu['sigma_r_deg']) ** 2
        self.Sigma_u = imu['q_scale'] * np.diag([var_a, var_a, var_r])
        p_floor = ekf['p_floor']
        self.Q_floor = np.diag([p_floor, p_floor, ekf['q_th'], ekf['q_v'], ekf['q_v']])
        self.gate = fid['gate']
        self.sigma_v0_sq = ekf['sigma_v0'] ** 2
        self.sigma_zv_sq = ekf.get('zupt_sigma_v', 0.005) ** 2
        self.q_board = np.array([ekf.get('zupt_q_p', 0.05) ** 2, ekf.get('zupt_q_p', 0.05) ** 2,
                                 math.radians(ekf.get('zupt_q_th_deg', 10.0)) ** 2])
        self.I5 = np.eye(5)
        self.F = np.eye(5)
        self.G = np.zeros((5, 3))

        self.x = np.zeros(5)
        self.P = np.zeros((5, 5))
        self.t = None
        self.last_fix = -math.inf
        self.initialised = False
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
        """Start the filter from the first accepted fix (report Section 2.5)."""
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
        """Propagate x and P to stamp in one step and buffer the result (report eq. 4)."""
        h = stamp - self.t
        self._step(u, h)
        if stopped:
            self.zero_velocity(h)
        self.buffer.push(stamp, self.x, self.P, u, h, stopped)
        self.t = stamp

    def zero_velocity(self, h):
        """End a step of h with the robot still: velocity zero, and the board free to move."""
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
        """Predict over a sample's interval, holding u; return the steps taken."""
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
        """Correct with a fix captured at stamp; return (accepted, nis) (report eqs. 7-8)."""
        b = self.buffer
        k = b.index_at(stamp)
        if k is None or stamp < self.last_fix:
            return False, math.nan
        nu, S = self._innovation_at(k, z, R)
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.gate:
            return False, nis
        K = np.linalg.solve(S, b.P[k, :3, :]).T
        self.x = b.x[k] + K @ nu
        self.x[2] = wrap_pi(self.x[2])
        IKH = self.I5.copy()
        IKH[:, :3] -= K
        P = IKH @ b.P[k] @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        self._replay_from(k)
        self.last_fix = stamp
        return True, nis

    def reanchor(self, z, R, stamp):
        """Restart the filter from a fix at its photo time, then replay the steps since."""
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
