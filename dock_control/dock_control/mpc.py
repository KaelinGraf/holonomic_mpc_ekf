"""
Docking MPC (report Section 3, brief Section 6): one QP per tick, without ROS.

What changes when:

* Constant, built once in Mpc.__init__: the weights as diagonal vectors, J^-,
  rows 0-199 of M (increments, amplitudes and wheel speeds act on dU through the
  constant C2 = L_Nc kron I3, so their left-hand sides never change) with the
  right-hand side of the increment rows, and the solver's set-up.
* Per tick: linearise -> augment -> predict_matrices, the cost, rows 200-319 of
  M (rows of Phi), the rest of gamma, then one solve.

The QP data live in dense working arrays (E_ext, A_ext, q_ext, l_ext, u_ext)
whose constant parts are written once; each tick writes only what changed.

Two solvers take the same arrays. DAQP (the default) is a dense dual active-set
solver: exact, about 0.15 ms a tick, and it never failed in the closed-loop
simulation. OSQP is the brief's choice and stays selectable, but ADMM converges
slowly on this small dense problem: near the dock it ran out of iterations on a
few per cent of ticks, each a fallback, even on QPs with no active constraint.
For OSQP the arrays are handed over at the fixed sparsity pattern's positions;
a scipy.sparse matrix is never rebuilt per tick, since its pattern would change
whenever an entry happened to be zero and update() would no longer match setup.
"""

from dataclasses import dataclass
import math

from dock_control import kinematics
import numpy as np
from scipy import sparse

# C_p = [0 I3]: the outputs are the pose half of x_p (report eq. 11).
CP = np.hstack([np.zeros((3, 3)), np.eye(3)])

SOLVERS = ('daqp', 'osqp')
# Brief Section 6, in osqp 1.x names (the Pi image pins 1.1.3, which does not polish
# by default); _setup_osqp renames them for osqp 0.6.
OSQP_SETTINGS = {
    'eps_abs': 1e-4, 'eps_rel': 1e-4, 'max_iter': 4000,
    'polishing': True, 'warm_starting': True, 'verbose': False,
}
OSQP_SOLVED = (1, 2)        # status_val: solved, solved inaccurate
DAQP_SOLVED = (1, 2)        # exitflag: optimal, soft optimal
INF = 1e30                  # what both solvers read as no bound
SLACK_WEIGHT = 2e2          # E_ext = blkdiag(E, 2e2) (brief Section 6, step 6)

# Amplitude rows for one step u = [u_b, v_b, r_b], in two shapes the supervisor
# picks per stage (Output.l1_cap), then |r_b| <= r_max in both:
# - per axis, |u_b| <= v_max and |v_b| <= v_max: APPROACH. The user's choice
#   (2026-10-14) over the brief's L1 cap, under which the plan spent the whole
#   cap on one axis, strafing at full speed with no forward motion.
# - L1, the four facets of |u_b| + |v_b| <= v_max: FINAL and HOLD. Per axis there,
#   the robot could strafe at the full v_final while driving in, and with the
#   wheels' lag it hunted sideways at the dock in simulation from every real
#   start of 2026-10-13 (mpc_policy_study in the report notes).
# The wheel-speed rows (w_max) bound a diagonal move under either.
A_AMP_AXIS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                       [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
A_AMP_L1 = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])


def f_c(x_c, u, Ts):
    """Return the pose kinematics of report eq. 9 for x_c = [x, y, theta]."""
    c, s = math.cos(x_c[2]), math.sin(x_c[2])
    return x_c + Ts * np.array((u[0] * c - u[1] * s, u[0] * s + u[1] * c, u[2]))


def linearise(x_c, u, Ts):
    """
    Return (A_k, B_k), 3x3 each, at the estimate and the last command (report eq. 10).

    mpc_linearise.m is the reference; both are held over the horizon.
    """
    c, s = math.cos(x_c[2]), math.sin(x_c[2])
    A = np.eye(3)
    A[0, 2] = -Ts * (u[0] * s + u[1] * c)
    A[1, 2] = Ts * (u[0] * c - u[1] * s)
    B = Ts * np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return A, B


def augment(A, B):
    """Return (Ap, Bp), 6x6 and 6x3, of the incremental model x_p = [dx_c; x_c] (report eq. 11)."""
    Ap = np.zeros((6, 6))
    Ap[:3, :3] = A
    Ap[3:, :3] = A
    Ap[3:, 3:] = np.eye(3)
    return Ap, np.vstack((B, B))


def predict_matrices(Ap, Bp, Np, Nc):
    """
    Return F (3Np x 6) and Phi (3Np x 3Nc), with Y = F x_p + Phi dU (report eq. 13).

    mpc_predict_matrices.m is the reference. Built with the recursion, not
    matrix powers: C_p Ap^(i+1) is one product from C_p Ap^i, and Phi is block
    Toeplitz ([Phi]_ij = C_p Ap^(i-j) Bp depends only on i - j), so the Np
    blocks are formed once and copied down the block columns.
    """
    F = np.empty((3 * Np, 6))
    Phi = np.zeros((3 * Np, 3 * Nc))
    blocks = np.empty((Np, 3, 3))
    CA = CP                             # C_p Ap^0
    for i in range(Np):
        blocks[i] = CA @ Bp             # C_p Ap^i Bp
        CA = CA @ Ap
        F[3 * i:3 * i + 3] = CA         # C_p Ap^(i+1)
    for j in range(Nc):
        Phi[3 * j:, 3 * j:3 * j + 3] = blocks[:Np - j].reshape(-1, 3)
    return F, Phi


def _pattern(rows, cols, shape):
    """Return a CSC matrix of ones on (rows, cols), and those positions in its data order."""
    M = sparse.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    M.sort_indices()
    return M, M.indices.copy(), np.repeat(np.arange(shape[1]), np.diff(M.indptr))


@dataclass
class MpcResult:
    """One tick's output (brief Section 6, step 7)."""

    u: np.ndarray           # body-velocity command [u_b, v_b, r_b]
    du: np.ndarray          # the applied increment
    dU: np.ndarray          # the whole planned increment sequence (3Nc)
    y_pred: np.ndarray      # Np x 3 predicted outputs, Y = F x_p + Phi dU
    slack: float
    exit_flag: int          # the solver's own code: DAQP exitflag or OSQP status_val
    fallback: bool


class Mpc:
    """Constants, the solver, and the per-tick step."""

    def __init__(self, p, solver='daqp'):
        if solver not in SOLVERS:
            raise ValueError(f'qp solver {solver!r} is not one of {SOLVERS}')
        mpc, lim, g = p['mpc'], p['lim'], p['geom']

        # --- constant, built once ---
        self.Ts, self.Np, self.Nc = mpc['Ts'], mpc['Np'], mpc['Nc']
        n = self.n = 3 * self.Nc                        # decision variables dU
        self.rho_s = mpc['rho_s']
        self.x_max = mpc['contact_band']                # no-penetration plane [m]
        self.th_max = math.radians(mpc['th_max_deg'])   # heading corridor [rad]
        self.w_max = lim['w_max']
        # du_max = T_s [a_max, a_max, alpha_max] (report eq. 15)
        self.du_max = self.Ts * np.array([lim['a_max'], lim['a_max'], lim['alpha_max']])
        # Q_bar = I_Np kron diag(q_w) Q_y and R_bar = I_Nc kron diag(r_w) R_u are
        # diagonal: keep the diagonals, so Phi' Q_bar Phi is Phi' (q_bar[:, None] * Phi).
        # q_w and r_w are tuning multipliers per axis on the derived weights: q_w on
        # the tracking of [x, y, theta], r_w on the effort of [u_b, v_b, r_b] (a scalar
        # applies to all three; q_w defaults to 1).
        self.Qy = np.asarray(mpc['Qy'], dtype=float)
        self.q_w = np.broadcast_to(np.asarray(mpc.get('qw', 1.0), dtype=float), (3,)).copy()
        self.q_bar = np.tile(self.q_w * self.Qy, self.Np)
        self.r_bar = np.tile(np.asarray(mpc['rw'], dtype=float)
                             * np.asarray(mpc['Ru'], dtype=float), self.Nc)
        self.J_inv = kinematics.ik_matrix(g['Rw'], g['lx'], g['ly'])
        self.A_wheel = np.vstack((self.J_inv, -self.J_inv))     # |J^- u| <= w_max, one step
        self.M_const = self._constant_rows(A_AMP_AXIS)
        # Row blocks of M: increments, amplitudes, wheel speeds, then the soft
        # output rows (no penetration, heading corridor up, heading corridor down).
        self.amp = slice(2 * n, 2 * n + 6 * self.Nc)
        # the amplitude block in either shape; A_ext holds the one step() was last asked for
        self.M_amp = {False: self.M_const[self.amp].copy(),
                      True: self._constant_rows(A_AMP_L1)[self.amp]}
        self.l1 = False
        self.wheel = slice(self.amp.stop, self.M_const.shape[0])
        k = self.wheel.stop
        self.pen = slice(k, k + self.Np)
        self.hup = slice(self.pen.stop, self.pen.stop + self.Np)
        self.hdn = slice(self.hup.stop, self.hup.stop + self.Np)
        m = self.m = self.hdn.stop                      # 20 Nc + 3 Np = 320 rows

        # The QP of brief Section 6 in dense working arrays, constant parts filled in:
        # min 1/2 z' E_ext z + q_ext' z  s.t.  l_ext <= A_ext z <= u_ext,  z = [dU; s].
        self.E_ext = np.zeros((n + 1, n + 1))
        self.E_ext[n, n] = SLACK_WEIGHT
        self.q_ext = np.zeros(n + 1)
        self.q_ext[n] = self.rho_s
        self.A_ext = np.zeros((m + 1, n + 1))
        self.A_ext[:k, :n] = self.M_const
        self.A_ext[k:m, n] = -1.0                       # the soft rows give way by s
        self.A_ext[m, n] = 1.0                          # s >= 0
        self.l_ext = np.full(m + 1, -INF)
        self.l_ext[m] = 0.0
        self.u_ext = np.full(m + 1, INF)
        self.u_ext[:2 * n] = np.minimum(np.tile(self.du_max, 2 * self.Nc), INF)  # constant
        self.r_diag = np.diag_indices(n)

        self.solver_name = solver
        if solver == 'daqp':
            import daqp     # here rather than at the top, so the maths imports without it
            self.daqp = daqp
        else:
            # OSQP gets the objective divided by rho_s, which leaves the minimiser
            # unchanged. Unscaled, the exact-penalty slack needs a multiplier near
            # rho_s (5.8e6) that ADMM approaches far too slowly to converge in
            # max_iter: OSQP clamps its own cost scaling at 1e-4.
            self.kappa = 1.0 / self.rho_s
            self._setup_osqp()

        # --- state: the last solution [dU; s], for OSQP's warm start ---
        self.z_prev = None

    def _constant_rows(self, a_amp):
        """
        Return rows 0-199 of M, the 200 x 3Nc block whose left-hand side never changes.

        With C1 = 1_Nc kron I3 and C2 = L_Nc kron I3 (L lower-triangular ones),
        U = C1 u(k-1) + C2 dU (report Section 3.4):

            increments    |du(k+j)| <= du_max                    6 Nc rows
            amplitudes    a_amp: |u_b|, |v_b| <= v_max per axis
                          or |u_b| + |v_b| <= v_max (L1), and
                          |r_b| <= r_max, on C2 dU               6 Nc rows
            wheel speeds  |(I_Nc kron J^-) C2 dU| <= w_max       8 Nc rows

        Only their right-hand sides change per tick (u(k-1), and the caps from
        the supervisor), and the amplitude block when the supervisor switches its
        shape. mpc_constraints.m is the reference (L1 amplitudes).
        """
        Nc = self.Nc
        C2 = np.kron(np.tril(np.ones((Nc, Nc))), np.eye(3))
        I_n = np.eye(3 * Nc)
        return np.vstack((I_n, -I_n,
                          np.kron(np.eye(Nc), a_amp) @ C2,
                          np.kron(np.eye(Nc), self.A_wheel) @ C2))

    def _setup_osqp(self):
        """
        Create the one osqp.OSQP() with the fixed sparsity pattern (brief Section 6).

        P is the upper triangle of E_ext: E is dense, plus the slack's diagonal.
        A is A_ext: the nonzeros of M_const (of either amplitude shape, so a switch
        keeps the pattern), the output rows taken as dense in dU, and the slack
        column. Set up once with placeholder ones (a positive semidefinite P); the
        index arrays, in CSC data order, let each tick hand OSQP the new values:
        update(Px=E_ext[P_rows, P_cols], Ax=A_ext[A_rows, A_cols]).
        """
        import osqp     # here rather than at the top, so that the maths imports without it

        n, m, k = self.n, self.m, self.wheel.stop
        P_rows, P_cols = np.triu_indices(n)
        P_rows, P_cols = np.append(P_rows, n), np.append(P_cols, n)
        union = self.M_const != 0.0
        union[self.amp] = (self.M_amp[False] != 0.0) | (self.M_amp[True] != 0.0)
        c_rows, c_cols = np.nonzero(union)
        o_rows, o_cols = np.divmod(np.arange((m - k) * n), n)
        A_rows = np.concatenate((c_rows, k + o_rows, np.arange(k, m + 1)))
        A_cols = np.concatenate((c_cols, o_cols, np.full(m + 1 - k, n)))
        P, self.P_rows, self.P_cols = _pattern(P_rows, P_cols, (n + 1, n + 1))
        A, self.A_rows, self.A_cols = _pattern(A_rows, A_cols, (m + 1, n + 1))
        settings, self.solve_kwargs = dict(OSQP_SETTINGS), {'raise_error': False}
        if int(osqp.__version__.split('.')[0]) < 1:     # osqp 0.6: older names, no raise_error
            settings['polish'] = settings.pop('polishing')
            settings['warm_start'] = settings.pop('warm_starting')
            self.solve_kwargs = {}
        self.solver = osqp.OSQP()
        self.solver.setup(P=P, q=self.kappa * self.q_ext, A=A, l=self.l_ext, u=self.u_ext,
                          **settings)

    def reset(self):
        """Forget the warm start, for a new run from rest."""
        self.z_prev = None

    def constraints(self, Phi, Fx, u_prev, v_max, r_max, v_lat_max=None):
        """
        Write this tick's M (320 x 3Nc) and gamma (320) and return them (report Section 3.4).

        Rows 0-199 are M_const; their right-hand sides are the caps less what
        u(k-1) already uses. Rows 200-319 are rows of Phi, so they change every
        tick:

            no penetration    x(k+j) <= x_max           Np rows
            heading corridor  |theta(k+j)| <= th_max    2 Np rows

        Fx = F x_p. Both output groups are soft (the slack column of A_ext). The
        two arrays are views into the solver's working copies: valid until the
        next tick.
        """
        n, Nc, A, u = self.n, self.Nc, self.A_ext, self.u_ext
        v_lat = v_max if v_lat_max is None or self.l1 else v_lat_max
        caps = np.array((v_max, v_max, v_lat, v_lat, r_max, r_max))  # L1: all four facets v_max
        a_amp = A_AMP_L1 if self.l1 else A_AMP_AXIS
        u[self.amp] = np.tile(caps - a_amp @ u_prev, Nc)
        u[self.wheel] = np.tile(self.w_max - self.A_wheel @ u_prev, Nc)
        Phi_th = Phi[2::3]
        A[self.pen, :n] = Phi[0::3]
        A[self.hup, :n] = Phi_th
        A[self.hdn, :n] = -Phi_th
        u[self.pen] = self.x_max - Fx[0::3]
        u[self.hup] = self.th_max - Fx[2::3]
        u[self.hdn] = self.th_max + Fx[2::3]
        np.minimum(u, INF, out=u)                       # a limit set to inf means no limit
        return A[:self.m, :n], u[:self.m]

    def step(self, x_hat, r_gyro, u_prev, y_ref, v_max, r_max, qw=None, l1_cap=False,
             v_lat_max=None):
        """
        Run one tick and return an MpcResult (brief Section 6, steps 1-8).

        1. x_p = [T_s [vx, vy, r_gyro]; [x, y, theta]] from x_hat (report eq. 12).
        2. A_k, B_k = linearise(x_hat[:3], u_prev).
        3. F, Phi = predict_matrices(augment(A_k, B_k)).
        4. E = 2 (Phi' Q_bar Phi + R_bar), f = -2 Phi' Q_bar (Y_r - F x_p), with
           Y_r the reference tiled Np times.
        5. M, gamma = constraints(...).
        6. z = [dU; s]: E_ext = blkdiag(E, 2e2), f_ext = [f; rho_s], -1 in the
           slack column of the soft rows, s >= 0 (fixed in __init__).
        7. Solve (_solve). On success: du = dU[0:3], u = u_prev + du,
           y_pred = F x_p + Phi dU as Np x 3.
        8. Otherwise fall back: decelerate each axis towards zero by its largest
           increment, and set fallback.

        qw, if given, replaces the tracking multipliers [x, y, theta] for this tick
        and those after it (the supervisor's per-stage schedule: FINAL locks the
        heading); Q_bar is only rebuilt when it changes. l1_cap picks the speed
        cap's shape, |u_b| + |v_b| <= v_max if true, else each axis on its own (the
        supervisor's: L1 in FINAL and HOLD); the amplitude block is only swapped
        when it changes. v_lat_max, if given, caps |v_b| apart from |u_b| under the per-axis
        rows (FINAL's straight-in: the robot must not strafe at the coupler).
        """
        n, Np, Nc, Ts = self.n, self.Np, self.Nc, self.Ts
        if qw is not None and not np.array_equal(qw, self.q_w):
            self.q_w = np.array(qw, dtype=float)
            self.q_bar = np.tile(self.q_w * self.Qy, Np)
        if bool(l1_cap) != self.l1:
            self.l1 = bool(l1_cap)
            self.A_ext[self.amp, :n] = self.M_amp[self.l1]
        x_p = np.array((Ts * x_hat[3], Ts * x_hat[4], Ts * r_gyro, x_hat[0], x_hat[1], x_hat[2]))
        A, B = linearise(x_hat[:3], u_prev, Ts)
        F, Phi = predict_matrices(*augment(A, B), Np, Nc)
        Fx = F @ x_p

        QPhi = self.q_bar[:, None] * Phi                # Q_bar Phi
        E = 2.0 * (Phi.T @ QPhi)
        E[self.r_diag] += 2.0 * self.r_bar
        self.E_ext[:n, :n] = E
        self.q_ext[:n] = -2.0 * (QPhi.T @ (np.tile(y_ref, Np) - Fx))
        self.constraints(Phi, Fx, u_prev, v_max, r_max, v_lat_max)

        z, flag = self._solve()
        if z is not None:
            dU = z[:n]
            du = dU[:3].copy()
            return MpcResult(u=u_prev + du, du=du, dU=dU, y_pred=(Fx + Phi @ dU).reshape(Np, 3),
                             slack=float(z[n]), exit_flag=flag, fallback=False)

        self.reset()
        du = -np.sign(u_prev) * np.minimum(np.abs(u_prev), self.du_max)
        dU = np.zeros(n)
        dU[:3] = du
        return MpcResult(u=u_prev + du, du=du, dU=dU, y_pred=(Fx + Phi @ dU).reshape(Np, 3),
                         slack=math.nan, exit_flag=flag, fallback=True)

    def _solve(self):
        """
        Solve this tick's QP; return (z, exit flag), z None unless solved.

        DAQP takes the dense arrays as they are. OSQP takes the values at its
        pattern's positions, with the objective scaled by kappa, warm-started
        from the last solution moved one step on (brief Section 6); status_val 1
        (solved) and 2 (solved inaccurate) are accepted.
        """
        if self.solver_name == 'daqp':
            z, _, flag, _ = self.daqp.solve(self.E_ext, self.q_ext, self.A_ext,
                                            self.u_ext, self.l_ext)
            ok = flag in DAQP_SOLVED
        else:
            self.solver.update(Px=self.kappa * self.E_ext[self.P_rows, self.P_cols],
                               Ax=self.A_ext[self.A_rows, self.A_cols],
                               q=self.kappa * self.q_ext, u=self.u_ext)
            if self.z_prev is not None:
                n = self.n
                z0 = np.zeros(n + 1)
                z0[:n - 3] = self.z_prev[3:n]
                z0[n] = self.z_prev[n]
                self.solver.warm_start(x=z0)
            res = self.solver.solve(**self.solve_kwargs)
            z, flag = res.x, res.info.status_val
            ok = flag in OSQP_SOLVED
        if not ok or z is None or not np.all(np.isfinite(z)):
            return None, int(flag)
        self.z_prev = np.array(z, dtype=float)
        return self.z_prev, int(flag)
