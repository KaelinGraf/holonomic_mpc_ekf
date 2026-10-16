function out = r2_tests()
%R2_TESTS Unit tests under the roll-back filter: Jacobians, noise-free tracking, target
%   velocity steps at 0.05 and 0.2 m/s, and the late-fix exactness check (Table F3).
prm = dock_params('fullscale');  T = prm.imu.T;
%% Jacobians against central differences
x0 = [-1.2; 0.3; 0.4; 0.15; -0.05];  u0 = [0.3; -0.2; 0.1];  P0 = 1e-3*eye(5);
[~, ~, F, G] = ekf_predict(x0, P0, u0, T, prm.ekf.Sigma_u, prm.ekf.Qfloor);
fx = @(x, u) ekf_predict(x, P0, u, T, prm.ekf.Sigma_u, prm.ekf.Qfloor);  h = 1e-6;
Fn = zeros(5);  Gn = zeros(5, 3);
for i = 1:5, e = zeros(5, 1); e(i) = h; d = fx(x0 + e, u0) - fx(x0 - e, u0); d(3) = wrap_pi(d(3)); Fn(:, i) = d/(2*h); end
for i = 1:3, e = zeros(3, 1); e(i) = h; d = fx(x0, u0 + e) - fx(x0, u0 - e); d(3) = wrap_pi(d(3)); Gn(:, i) = d/(2*h); end
out.jacobian_F_err = max(abs(F(:) - Fn(:)));  out.jacobian_G_err = max(abs(G(:) - Gn(:)));
%% chain Jacobian against central differences
g = prm.geom;  m0 = [0.7; -0.1; deg2rad(5)];  [~, J] = fid_chain(m0, g);  Jn = zeros(3);
for i = 1:3, e = zeros(3, 1); e(i) = h; d = fid_chain(m0 + e, g) - fid_chain(m0 - e, g); d(3) = wrap_pi(d(3)); Jn(:, i) = d/(2*h); end
out.chain_J_err = max(abs(J(:) - Jn(:)));
%% noise-free tracking
tr = r1_traj(prm);
o = run_ekf(tr, prm, struct('seed', 1, 'noise_scale', 0, 'bias', false));
m = ~isnan(o.err(:, 1));
out.noise_free_pos_mm = 1e3*max(hypot(o.err(m, 1), o.err(m, 2)));
out.noise_free_th_deg = rad2deg(max(abs(o.err(m, 3))));
out.noise_free_v_mms = 1e3*max(hypot(o.err(m, 4), o.err(m, 5)));
%% target velocity step (the robot's IMU cannot see it)
t_step = 6;  o0 = run_ekf(tr, prm, struct('seed', 7));
after = tr.t > t_step + 2.5 & tr.t < t_step + 4;  w = round(0.5/T);   % the board leaves the 1.45 m range later
ev0 = hypot(o0.err(:, 4), o0.err(:, 5));  rms0 = rms(hypot(o0.err(after, 1), o0.err(after, 2)));
k = 0;
for dv = [0.05, 0.20]
    k = k + 1;  dvec = [dv; 0.3*dv];        % target moving away and sideways: the board stays in view
    trs = apply_target_velocity_step(tr, t_step, dvec, 0.5);
    o = run_ekf(trs, prm, struct('seed', 7));
    ev = hypot(o.err(:, 4), o.err(:, 5));  ep = hypot(o.err(:, 1), o.err(:, 2));
    thr = max(0.1*norm(dvec), 2*rms(ev0(after)));
    outside = find(tr.t > t_step & tr.t < t_step + 4 & movmean(ev, w) >= thr);
    if isempty(outside), t_rec = 0; else, t_rec = tr.t(outside(end)) - t_step; end
    out.target_step(k) = struct('dv_mps', dv, 'recovery_s', t_rec, ...
        'peak_pos_err_mm', 1e3*max(ep(tr.t >= t_step & tr.t < t_step + 4)), ...
        'rms_after_mm', 1e3*rms(ep(after)), 'rms_baseline_mm', 1e3*rms0, 'rejected', o.n_rej, 'reanchors', o.n_reanchor);
end
%% late fix: a fix applied 147 ms late leaves the state an on-time one would
f = ekf_create(prm);  f = ekf_fix(f, [-0.8; 0.1; 0.05], diag([1e-4 1e-4 1e-4]), 0, 'init');
rs = RandStream('mt19937ar', 'Seed', 3);  n = 60;  U = [0.2*randn(rs, 2, n); 0.1*randn(rs, 1, n)];
tc = 10*T;  z = [-0.79; 0.105; 0.052];  Rz = diag([2e-5 2e-5 3e-4]);
fa = f;  fb = f;
for i = 1:n
    fa = ekf_advance(fa, i*T, U(:, i), false);
    if abs(i*T - tc) < 1e-12, fa = ekf_fix(fa, z, Rz, tc, 'gated'); end      % on time
    fb = ekf_advance(fb, i*T, U(:, i), false);
    if abs(i*T - (tc + 0.147)) < T/2, fb = ekf_fix(fb, z, Rz, tc, 'gated'); end   % 147 ms late
end
out.late_fix_dx = max(abs(fa.x - fb.x));  out.late_fix_dP = max(abs(fa.P(:) - fb.P(:)));
end
