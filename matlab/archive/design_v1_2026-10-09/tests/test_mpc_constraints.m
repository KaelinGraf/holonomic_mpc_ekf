function r = test_mpc_constraints()
%TEST_MPC_CONSTRAINTS The constrained tick respects every limit over the horizon.
%   From 0.5 m out, already moving at the approach speed limit, with the no-penetration
%   plane at x = 0: the solution must keep every increment, command, wheel speed and
%   predicted x inside its limit, and the amplitude limit must actually be active
%   (the unconstrained solution would exceed it). The speed cap is the L1 bound
%   |u_b| + |v_b| <= v_max. Also checks the row count 20*Nc + 3*Np.
%   Catches: constraint blocks in the wrong order or sign, C1/C2 built wrongly,
%   wheel rows using the forward instead of the inverse kinematics, penetration rows
%   missing F*xp on the right-hand side.
r.name = 'MPC constrained tick (increments, amplitudes, wheels, no-penetration)';
prm = dock_params('fullscale');
lim = prm.mpc.lim;
Ts = prm.mpc.Ts;  Np = prm.mpc.Np;  Nc = prm.mpc.Nc;
xhat = [-0.5; 0.02; 0.05; 0.30; 0.0];  r_gyro = 0;  u_prev = [0.30; 0; 0];  yr = zeros(3,1);
[u, info] = mpc_step(xhat, r_gyro, u_prev, yr, lim, prm);
dU = reshape(info.dU, 3, Nc);
U  = u_prev + cumsum(dU, 2);
[~, Jinv] = mecanum_ik(zeros(3,1), prm.geom);
W  = Jinv*U;
tol = 1e-6;
ok_du = all(abs(dU(:)) <= repmat(lim.du_max, Nc, 1) + tol);
ok_u  = all(abs(U(1,:)) + abs(U(2,:)) <= lim.v_max + tol) && all(abs(U(3,:)) <= lim.r_max + tol);
ok_w  = all(abs(W(:)) <= lim.w_max + tol);
ok_x  = all(info.Y(:,1) <= lim.x_max + tol) && all(abs(info.Y(:,3)) <= lim.th_max + tol);
% row count of the assembled constraint set
xp = [Ts*[xhat(4); xhat(5); r_gyro]; xhat(1:3)];
[A, B] = mpc_linearise(xhat(1:3), u_prev, Ts);
[F, Phi] = mpc_predict_matrices(A, B, Np, Nc);
[M, gam] = mpc_constraints(F, Phi, xp, u_prev, lim, Jinv, Nc, Np);
ok_rows = isequal(size(M), [20*Nc + 3*Np, 3*Nc]) && numel(gam) == 20*Nc + 3*Np;
ok_feas = all(M*info.dU <= gam + tol) && ~info.fallback;
% the amplitude limit must bind: the unconstrained tick from this state would exceed it
lim_free = lim;  lim_free.v_max = 1e3;  lim_free.r_max = 1e3;  lim_free.du_max = 1e3*[1;1;1];  lim_free.w_max = 1e6;  lim_free.x_max = 1e3;  lim_free.th_max = 1e3;
u_free = mpc_step(xhat, r_gyro, u_prev, yr, lim_free, prm);
binding = abs(u_free(1)) + abs(u_free(2)) > lim.v_max + 1e-3;
r.pass = ok_du && ok_u && ok_w && ok_x && ok_rows && ok_feas && binding;
r.warn = false;
r.msg  = sprintf('increments %d, amplitudes %d, wheels %d, no-penetration %d, rows %d, M*dU<=gam %d, limit binding %d; u = [%.3f %.3f %.3f], max predicted x = %.4f', ...
                 ok_du, ok_u, ok_w, ok_x, ok_rows, ok_feas, binding, u, max(info.Y(:,1)));
end
