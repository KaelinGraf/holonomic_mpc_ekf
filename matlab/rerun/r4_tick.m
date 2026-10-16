function out = r4_tick()
%R4_TICK Constrained tick at N_p = 80 with per-axis caps, 0.5 m out at the cap, both scales.
k = 0;
for v = {'fullscale', 'demo'}
    k = k + 1;  prm = dock_params(v{1});
    va = prm.cap.v_approach;  xhat = [-0.5; 0.02; 0.05; va; 0];  u_prev = [va; 0; 0];
    lim = struct('du_max', prm.mpc.du_max, 'v_max', va, 'v_lat', va, 'l1', false, 'r_max', prm.lim.r_max, ...
                 'w_max', prm.lim.w_max, 'x_max', prm.mpc.x_max, 'th_max', prm.mpc.th_max);
    [u, info] = mpc_step(xhat, 0, u_prev, zeros(3, 1), lim, prm, prm.mpc.qw_approach_near);
    Nc = prm.mpc.Nc;  dU = reshape(info.dU, 3, Nc);  U = u_prev + cumsum(dU, 2);  W = prm.geom.Jinv*U;  tol = 1e-6;
    ok_du = all(abs(dU) <= prm.mpc.du_max + tol, 'all');
    ok_u = all(abs(U(1:2, :)) <= va + tol, 'all') && all(abs(U(3, :)) <= prm.lim.r_max + tol);
    ok_w = all(abs(W) <= prm.lim.w_max + tol, 'all');
    ok_x = all(info.Y(:, 1) <= prm.mpc.x_max + max(info.slack, 0) + tol);
    ok_M = all(info.M*info.dU <= info.gam + max(info.slack, 0) + tol);
    out(k) = struct('variant', v{1}, 'rows', info.rows, 'rows_expected', 20*Nc + 3*prm.mpc.Np, ...
        'increments_ok', ok_du, 'amplitudes_ok', ok_u, 'wheels_ok', ok_w, 'penetration_ok', ok_x, 'all_rows_ok', ok_M, ...
        'u', u', 'max_pred_x_m', max(info.Y(:, 1)), 'x_max_m', prm.mpc.x_max, 'slack', info.slack, ...
        'exitflag', info.exitflag, 'fallback', info.fallback); %#ok<AGROW>
end
end
