function [u, info] = mpc_step(xhat, r_gyro, u_prev, yr, lim, prm)
%MPC_STEP One tick of the successive-linearisation MPC.
%   [u, info] = MPC_STEP(xhat, r_gyro, u_prev, yr, lim, prm)
%   xhat    [5x1]  EKF estimate [x; y; th; vx; vy] in the target frame
%   r_gyro  current yaw rate from the gyro [rad/s]
%   u_prev  [3x1]  last applied body-velocity command
%   yr      [3x1]  pose reference held over the horizon (standoff pose or the origin)
%   lim     constraint limits for this stage (prm.mpc.lim, with v_max lowered by the supervisor
%           in the final approach)
%   prm     from dock_params (uses prm.mpc.Ts, Np, Nc, Qy, Ru, rw and prm.geom)
%   u       [3x1]  command to apply now, u = u_prev + du(k)
%   info    struct, at least: dU [3Nc x 1], Y [Np x 3] predicted outputs, xp [6x1],
%           slack (the penetration slack s), exitflag from the solver,
%           fallback (logical: solver failed, used the stop action)
%
%   Steps (design doc Sections 4.2-4.7):
%     x_p = [Ts*[vx_hat; vy_hat; r_gyro];  [x; y; th]_hat]            eq. (24)
%     [A, B] = mpc_linearise(xc, u_prev, Ts);  [F, Phi] = mpc_predict_matrices(A, B, Np, Nc)
%     Qbar = kron(eye(Np), Qy);  Rbar = rw * kron(eye(Nc), Ru);  Yr = kron(ones(Np,1), yr)
%     E = 2*(Phi'*Qbar*Phi + Rbar);  f = -2*Phi'*Qbar*(Yr - F*x_p)       eq. (38)
%     [M, gam] = mpc_constraints(...)                                      eq. (37)
%   The first 20*Nc rows of M (increments, amplitudes, wheels) are hard; they are always
%   feasible because dU = 0 satisfies them. The last 3*Np rows (no-penetration and the
%   heading corridor) are SOFT:
%   add one slack s >= 0 to the decision vector, subtract it from those rows, and add
%   the LINEAR penalty prm.mpc.rho_pen*s to the cost (course Section 5.4.1). A linear
%   penalty larger than any constraint multiplier is exact: s is zero whenever the hard
%   constraint is feasible, unlike a quadratic penalty, which always lets a small
%   violation through. A hard version is infeasible whenever the estimate is already past
%   the plane and the increment limits forbid an instant stop. Solve with quadprog on
%   z = [dU; s] (lower bound s >= 0; a tiny quadratic term on s keeps E positive definite).
%   Solver note: quadprog's 'active-set' algorithm with the feasible start z0 = [0; s0],
%   s0 = max(0, max of the penetration rows at dU = 0), is reliable for this small dense
%   QP; the interior-point default occasionally fails on the near-degenerate ticks at
%   contact, while active-set can stall (exitflag 0) on a degenerate vertex, e.g. driving
%   at the vertex of the L1 cap with two wheels at their limit. So: active-set first,
%   retry with interior-point-convex if it reports failure, safe stop only if both fail.
%   Then
%     u = u_prev + dU(1:3)
%   If the solver still fails, apply the safe stop action: decelerate every axis by the
%   largest allowed increment towards zero, and flag info.fallback.
%   Checked by tests/test_mpc_unconstrained (closed form), tests/test_mpc_constraints
%   and the closed-loop tests.

Ts = prm.mpc.Ts;  Np = prm.mpc.Np;  Nc = prm.mpc.Nc;
xc = xhat(1:3);
xp = [Ts*[xhat(4); xhat(5); r_gyro]; xc(:)];

[A, B]   = mpc_linearise(xc, u_prev, Ts);
[F, Phi] = mpc_predict_matrices(A, B, Np, Nc);
Qbar = kron(eye(Np), prm.mpc.Qy);
Rbar = prm.mpc.rw * kron(eye(Nc), prm.mpc.Ru);
Yr   = kron(ones(Np, 1), yr(:));
E = 2*(Phi'*Qbar*Phi + Rbar);  E = (E + E')/2;
f = -2*Phi'*Qbar*(Yr - F*xp);

[~, Jinv] = mecanum_ik(zeros(3, 1), prm.geom);
[M, gam]  = mpc_constraints(F, Phi, xp, u_prev, lim, Jinv, Nc, Np);
nh = 20*Nc;                                   % hard rows: increments, amplitudes, wheels
ns = size(M, 1) - nh;                         % soft rows: no-penetration and heading corridor
nz = 3*Nc;
% extended decision vector z = [dU; s]: linear (exact) penalty on s, tiny quadratic for conditioning
E_ext = blkdiag(E, 2*1e2);
f_ext = [f; prm.mpc.rho_pen];
M_ext = [M(1:nh, :), zeros(nh, 1);  M(nh+1:end, :), -ones(ns, 1)];
lb    = [-inf(nz, 1); 0];

% active-set with a feasible start (dU = 0, just enough slack) is reliable for this small
% dense QP, but it can stall (exitflag 0) on a degenerate vertex, e.g. driving at the
% vertex of the L1 speed cap with two wheels at their limit. Second attempt with
% interior-point-convex, which handles degeneracy but occasionally reports -3 on the
% near-degenerate ticks at contact. Only both failing is a fallback.
persistent opts_as opts_ip
if isempty(opts_as)
    opts_as = optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'off', 'MaxIterations', 2000);
    opts_ip = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
end
s0 = max(0, max(M(nh+1:end, :)*zeros(nz, 1) - gam(nh+1:end)));
z0 = [zeros(nz, 1); s0];
[z, ~, exitflag] = quadprog(E_ext, f_ext, M_ext, gam, [], [], lb, [], z0, opts_as);
if exitflag <= 0 || isempty(z)
    [z, ~, exitflag] = quadprog(E_ext, f_ext, M_ext, gam, [], [], lb, [], [], opts_ip);
end
fallback = exitflag <= 0 || isempty(z);
if fallback
    % safe stop: decelerate every axis by the largest allowed increment towards zero
    du = -sign(u_prev(:)) .* min(abs(u_prev(:)), lim.du_max(:));
    dU = [du; zeros(3*(Nc-1), 1)];  s = NaN;
else
    dU = z(1:nz);  s = z(end);
end
du = dU(1:3);
u  = u_prev(:) + du;

info.dU = dU;
info.Y  = reshape(F*xp + Phi*dU, 3, Np)';
info.xp = xp;
info.slack = s;
info.exitflag = exitflag;
info.fallback = fallback;
info.active   = M(1:nh, :)*dU >= gam(1:nh) - 1e-9;
end
