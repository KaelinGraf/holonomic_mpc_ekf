function [u, info] = mpc_step(xhat, r_gyro, u_prev, yr, lim, prm, qw)
%MPC_STEP One tick of the deployed successive-linearisation MPC (mpc.py Mpc.step).
%   lim: du_max, v_max, v_lat, l1, r_max, w_max, x_max, th_max (from sup_step)
%   qw:  stage multipliers on Qy (APPROACH ramp, FINAL [1 1 10]); effort multipliers
%        prm.mpc.rw on Ru. One slack s >= 0 shared by the soft rows, cost rho_s*s + w_eps*s^2/2.
%   Solver: quadprog active-set from the feasible start, interior-point retry; a failed
%   solve decelerates each axis by its largest increment (fallback).
if nargin < 7 || isempty(qw), qw = [1; 1; 1]; end
Ts = prm.mpc.Ts;  Np = prm.mpc.Np;  Nc = prm.mpc.Nc;
xp = [Ts*[xhat(4); xhat(5); r_gyro]; xhat(1:3)];
[A, B] = mpc_linearise(xhat(1:3), u_prev, Ts);
[F, Phi] = mpc_predict_matrices(A, B, Np, Nc);
qbar = repmat(qw(:) .* diag(prm.mpc.Qy), Np, 1);
rbar = repmat(prm.mpc.rw(:) .* diag(prm.mpc.Ru), Nc, 1);
Yr = repmat(yr(:), Np, 1);
QPhi = qbar .* Phi;
E = 2*(Phi'*QPhi) + 2*diag(rbar);  E = 0.5*(E + E');
f = -2*QPhi'*(Yr - F*xp);
[M, gam] = mpc_constraints(F, Phi, xp, u_prev, lim, prm.geom.Jinv, Nc, Np);
nh = 20*Nc;  ns = size(M, 1) - nh;  nz = 3*Nc;
E_ext = blkdiag(E, prm.mpc.w_eps);
f_ext = [f; prm.mpc.rho_s];
M_ext = [M(1:nh, :), zeros(nh, 1); M(nh+1:end, :), -ones(ns, 1)];
lb = [-inf(nz, 1); 0];
persistent oa oi
if isempty(oa)
    oa = optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'off', 'MaxIterations', 2000);
    oi = optimoptions('quadprog', 'Algorithm', 'interior-point-convex', 'Display', 'off');
end
s0 = max(0, max(-gam(nh+1:end)));
[z, ~, ef] = quadprog(E_ext, f_ext, M_ext, gam, [], [], lb, [], [zeros(nz, 1); s0], oa);
if ef <= 0 || isempty(z)
    [z, ~, ef] = quadprog(E_ext, f_ext, M_ext, gam, [], [], lb, [], [], oi);
end
info.fallback = ef <= 0 || isempty(z);
if info.fallback
    du = -sign(u_prev(:)) .* min(abs(u_prev(:)), lim.du_max(:));
    dU = [du; zeros(3*(Nc-1), 1)];  info.slack = NaN;
else
    dU = z(1:nz);  info.slack = z(end);
end
u = u_prev(:) + dU(1:3);
info.dU = dU;  info.exitflag = ef;
info.Y = reshape(F*xp + Phi*dU, 3, Np)';
info.rows = size(M, 1);
info.M = M;  info.gam = gam;
end
