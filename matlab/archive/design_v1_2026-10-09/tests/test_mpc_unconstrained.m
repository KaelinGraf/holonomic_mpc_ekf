function r = test_mpc_unconstrained()
%TEST_MPC_UNCONSTRAINED With the limits removed, mpc_step must return the closed-form solution.
%   The course's eq. for dU = (Phi'Qbar Phi + Rbar)^-1 Phi'Qbar (Yr - F xp), MIMO version.
%   Also checks that u = u_prev + dU(1:3), that the predicted outputs are reported,
%   and that at zero heading the lateral command is zero for a purely longitudinal error.
%   Catches: cost assembled with the wrong weights, Yr not stacked over the horizon,
%   the whole dU applied instead of its first block, wrong sign in f.
r.name = 'MPC unconstrained tick equals the closed form';
prm = dock_params('fullscale');
lim = prm.mpc.lim;
lim.du_max = 1e3*[1;1;1];  lim.v_max = 1e3;  lim.r_max = 1e3;  lim.w_max = 1e6;  lim.x_max = 1e3;  lim.th_max = 1e3;
xhat = [-1.0; 0.0; 0.0; 0.0; 0.0];  r_gyro = 0;  u_prev = zeros(3,1);  yr = zeros(3,1);
[u, info] = mpc_step(xhat, r_gyro, u_prev, yr, lim, prm);
Ts = prm.mpc.Ts;  Np = prm.mpc.Np;  Nc = prm.mpc.Nc;
xp = [Ts*[xhat(4); xhat(5); r_gyro]; xhat(1:3)];
[A, B] = mpc_linearise(xhat(1:3), u_prev, Ts);
[F, Phi] = mpc_predict_matrices(A, B, Np, Nc);
Qbar = kron(eye(Np), prm.mpc.Qy);  Rbar = prm.mpc.rw*kron(eye(Nc), prm.mpc.Ru);
Yr = kron(ones(Np,1), yr);
dU_cf = (Phi'*Qbar*Phi + Rbar) \ (Phi'*Qbar*(Yr - F*xp));
err_dU = max(abs(info.dU - dU_cf)) / max(1e-9, max(abs(dU_cf)));
err_u  = max(abs(u - (u_prev + dU_cf(1:3))));
lateral_zero = abs(u(2)) < 1e-9 && abs(u(3)) < 1e-9;
forward = u(1) > 0;
r.pass = err_dU < 1e-6 && err_u < 1e-6 && lateral_zero && forward && isfield(info, 'Y') && isequal(size(info.Y), [Np, 3]);
r.warn = false;
r.msg  = sprintf('relative |dU - closed form| = %.1e, |u - u_cf| = %.1e, u = [%.3f %.3f %.3f]', err_dU, err_u, u);
end
