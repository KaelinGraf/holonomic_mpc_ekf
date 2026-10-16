function [u, info] = mpc_step(xhat, r_gyro, u_prev, yr, lim, prm)
%MPC_STEP [REFERENCE] One tick of the successive-linearisation MPC with quadprog.
%   The no-penetration rows are soft (one slack s >= 0 with a linear penalty rho_pen*s,
%   an exact penalty: s stays at zero whenever the hard constraint is feasible), as the
%   course recommends for output constraints; the increment, amplitude and wheel-speed
%   rows are hard and always feasible provided u_prev is inside the limits.
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
nh = 20*Nc;
ns = size(M, 1) - nh;
nz = 3*Nc;
E_ext = blkdiag(E, 2*1e2);
f_ext = [f; prm.mpc.rho_pen];
M_ext = [M(1:nh, :), zeros(nh, 1);  M(nh+1:end, :), -ones(ns, 1)];
lb    = [-inf(nz, 1); 0];

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
