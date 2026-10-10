"""
Docking EKF (report Section 2, brief Section 5), without ROS.

State x = [x, y, theta, vx, vy] in the target frame; input u = [a_x, a_y, r],
the body-frame linear acceleration (gravity removed, start-up offset subtracted)
and the yaw rate. predict() runs at 200 Hz, so everything that does not change
between samples is built once in __init__: the noise matrices, the ring buffer,
and F and G with their constant entries already set, which leaves the sample to
write six entries. H = [I3 0] is a selection, so update() slices instead of
multiplying: H x = x[:3], H P H' = P[:3, :3], P H' = P[:, :3].
"""

import math

from dock_control.geometry import wrap_pi
import numpy as np


class RingBuffer:
    """
    Estimates over the last buffer_s seconds, keyed by IMU stamp.

    Preallocated: push() overwrites the oldest slot, so nothing is allocated at
    200 Hz. A fix is about ten samples old when it arrives, so at() scanning back
    from the newest entry is quicker than a binary search.
    """

    def __init__(self, depth, n=5):
        self.depth = depth
        self.t = np.full(depth, -math.inf)     # stamps [s]; -inf marks an empty slot
        self.x = np.zeros((depth, n))
        self.head = -1                         # slot of the newest entry

    def push(self, stamp, x):
        """Store x at stamp in the oldest slot."""
        self.head = (self.head + 1) % self.depth
        self.t[self.head] = stamp
        self.x[self.head] = x

    def at(self, stamp):
        """
        Return the latest stored state with t <= stamp, or None if stamp predates the buffer.

        The row is a view into the buffer: use it before the next push().
        """
        i = self.head
        for _ in range(self.depth):
            t = self.t[i]
            if t <= stamp:
                return None if t == -math.inf else self.x[i]
            i = i - 1 if i else self.depth - 1
        return None

    def correct_from(self, stamp, dx):
        """
        Add a correction to every entry at or after stamp (brief Section 5).

        The brief's "store the post-correction estimate": a later fix captured in
        that window is then compared with a corrected state rather than
        correcting the same error twice, which matters once the pose delay
        exceeds the fix period (WiFi: 107 ms median against 67 ms). The entries
        are shifted, not re-propagated: the design accepts that second-order error.
        """
        i = self.head
        for _ in range(self.depth):
            if self.t[i] < stamp:
                return
            self.x[i] += dx
            i = i - 1 if i else self.depth - 1


class Ekf:
    """The filter: constants from the parameter file, then initialise, predict, update."""

    def __init__(self, p):
        imu, fid, ekf = p['imu'], p['fid'], p['ekf']

        # --- constant, built once ---
        self.T = 1.0 / imu['rate']
        var_a = imu['sigma_a'] ** 2
        var_r = math.radians(imu['sigma_r_deg']) ** 2
        # per-sample IMU noise inflated by q_scale, and the process-noise floor (report eq. 4)
        self.Sigma_u = imu['q_scale'] * np.diag([var_a, var_a, var_r])
        p_floor = ekf['p_floor']
        self.Q_floor = np.diag([p_floor, p_floor, ekf['q_th'], ekf['q_v'], ekf['q_v']])
        self.gate = fid['gate']
        self.sigma_v0_sq = ekf['sigma_v0'] ** 2
        self.I5 = np.eye(5)
        # F = I5 + T df/dx and G = T df/du (report eq. 3). The entries that never
        # change are set here; jacobians() writes F[3, 2], F[4, 2] and G[3:5, 0:2].
        self.F = np.eye(5)
        self.F[0, 3] = self.F[1, 4] = self.T
        self.G = np.zeros((5, 3))
        self.G[2, 2] = self.T

        # --- state ---
        self.x = np.zeros(5)
        self.P = np.zeros((5, 5))
        self.t = None               # the filter's clock: time of the last step [s]
        self.initialised = False
        self.buffer = RingBuffer(round(ekf['buffer_s'] * imu['rate']))

    def f(self, x, u):
        """Return the Euler step of report eq. 2 from (x, u), without touching the filter state."""
        c, s = math.cos(x[2]), math.sin(x[2])
        ax, ay, r = u
        return x + self.T * np.array((x[3], x[4], r, c * ax - s * ay, s * ax + c * ay))

    def jacobians(self, x, u):
        """
        Write the sample-dependent entries of self.F and self.G at (x, u); return (F, G).

        Report eq. 3, evaluated at the estimate before the step.
        """
        c, s = math.cos(x[2]), math.sin(x[2])
        ax, ay = u[0], u[1]
        T, F, G = self.T, self.F, self.G
        F[3, 2] = -T * (ax * s + ay * c)
        F[4, 2] = T * (ax * c - ay * s)
        G[3, 0], G[3, 1] = T * c, -T * s
        G[4, 0], G[4, 1] = T * s, T * c
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
        self.buffer.push(stamp, self.x)
        self.t = stamp
        self.initialised = True

    def predict(self, u, stamp):
        """
        Propagate x and P over one step of T and buffer the result at stamp (report eq. 4).

        x <- f(x, u) with the heading wrapped; P <- F P F' + G Sigma_u G' + Q_floor,
        symmetrised, with F and G evaluated before the step.
        """
        F, G = self.jacobians(self.x, u)
        self.x = self.f(self.x, u)
        self.x[2] = wrap_pi(self.x[2])
        P = F @ self.P @ F.T + G @ self.Sigma_u @ G.T + self.Q_floor
        self.P = 0.5 * (P + P.T)
        self.buffer.push(stamp, self.x)
        self.t = stamp

    def advance(self, u, stamp):
        """
        Predict in steps of T up to a sample's stamp, holding u; return the steps taken.

        The model is discretised at T (report eq. 2), so the filter keeps its own
        clock and takes round((stamp - t) / T) steps per sample: one at the
        IMU's 200 Hz (the MATLAB schedule exactly), more after a dropped sample
        or from a slower stream, none for a repeated stamp. Time is neither lost
        nor invented, whatever the IMU's rate. A gap longer than the buffer
        steps one buffer's worth and re-anchors the clock to the stamp.
        """
        n = round((stamp - self.t) / self.T)
        if n <= 0:
            return 0
        steps = min(n, self.buffer.depth)
        for _ in range(steps):
            self.predict(u, self.t + self.T)
        if n > steps:
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

        Innovation against the buffered estimate at the capture stamp, heading
        wrapped; rejected if nis = nu' S^-1 nu exceeds the gate or the stamp
        predates the buffer. The correction goes to the current state, heading
        wrapped, with the Joseph-form covariance; the buffer entries from the
        capture stamp on are shifted by the same correction.
        """
        innovation = self._innovation(z, R, stamp)
        if innovation is None:
            return False, math.nan
        nu, S = innovation
        nis = float(nu @ np.linalg.solve(S, nu))
        if nis > self.gate:
            return False, nis
        K = np.linalg.solve(S, self.P[:3, :]).T     # P H' S^-1, without an inverse
        dx = K @ nu
        self.x = self.x + dx
        self.x[2] = wrap_pi(self.x[2])
        IKH = self.I5.copy()
        IKH[:, :3] -= K                             # I - K H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)
        self.buffer.correct_from(stamp, dx)
        return True, nis

    def _innovation(self, z, R, stamp):
        x_ref = self.buffer.at(stamp)
        if x_ref is None:
            return None
        nu = z - x_ref[:3]
        nu[2] = wrap_pi(nu[2])
        return nu, self.P[:3, :3] + R
