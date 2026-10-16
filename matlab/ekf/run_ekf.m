function out = run_ekf(tr, prm, opts)
%RUN_EKF The deployed filter alone on an analytic truth trajectory (R1, R2, R3).
%   out = RUN_EKF(tr, prm, opts);  tr from make_truth_trajectory sampled at prm.imu.T,
%   starting with >= 2 s at rest (the accelerometer zeroing).
%   opts .seed, .noise_scale, .bias (true), .dropout [t1 t2] (fixes captured inside are lost),
%        .delay (default prm.fid.delay), .zv (true: zero-velocity reset while the truth
%        has been still for 0.3 s, standing in for zero commands)
%   out  .t, .x, .err (heading wrapped), .Pdiag, .nees, .nis (per fix), .nis_acc,
%        .fix_t, .n_acc, .n_rej, .t_start
if nargin < 3, opts = struct(); end
seed = getfield_default(opts, 'seed', 1);
ns = getfield_default(opts, 'noise_scale', 1);
use_bias = getfield_default(opts, 'bias', true);
dropout = getfield_default(opts, 'dropout', []);
delay = getfield_default(opts, 'delay', prm.fid.delay);
use_zv = getfield_default(opts, 'zv', true);
rs = RandStream('mt19937ar', 'Seed', seed);
N = numel(tr.t);  T = prm.imu.T;
imu_n = randn(rs, N, 3);
kc = unique(round((0:1/prm.fid.f:tr.t(end))/T)) + 1;  kc = kc(kc <= N);
fix_n = randn(rs, numel(kc), 3);
nd = ekf_node_create(prm);  f = ekf_create(prm);
xt = [tr.pose, tr.vel(:, 1:2)];
X = nan(N, 5);  Pd = nan(N, 5);  nees = nan(N, 1);
queue = [];  qz = {};  last_move = -inf;  fixes = zeros(0, 3);
for k = 1:N
    t = tr.t(k);
    if norm(tr.vel(k, :)) > prm.ekf.zv_cmd_eps, last_move = t; end
    a = tr.ab(k, :)' + use_bias*(prm.imu.bias_rest_a + (t >= prm.sim.t_zero)*prm.imu.bias_a) + ns*prm.imu.sigma_a*imu_n(k, 1:2)';
    r = tr.r(k) + use_bias*prm.imu.bias_r + ns*prm.imu.sigma_r*imu_n(k, 3);
    if ~nd.zeroed
        if isempty(nd.zero_t0), nd.zero_t0 = t; end
        nd.zero_sum = nd.zero_sum + a;  nd.zero_n = nd.zero_n + 1;
        if t - nd.zero_t0 >= nd.t_zero, nd.offset = nd.zero_sum/nd.zero_n; nd.zeroed = true; end
    else
        f = ekf_advance(f, t, [a - nd.offset; r], use_zv && t - last_move >= prm.ekf.zv_quiet);
    end
    i = find(kc == k, 1);
    if ~isempty(i)
        fx = fiducial_measure(tr.pose(k, :)', prm, fix_n(i, :)', ns);
        lost = ~isempty(dropout) && t >= dropout(1) && t < dropout(2);
        if fx.valid && ~lost, queue(end+1, :) = [t, t + delay]; qz{end+1} = fx; end
    end
    while ~isempty(queue) && queue(1, 2) <= t + 1e-9
        fx = qz{1};  fx.t = queue(1, 1);
        if nd.zeroed && fx.t >= nd.zero_t0 + nd.t_zero
            n0 = nd.n_acc;
            [f, nd] = ekf_node_fix(f, nd, fx, false);
            fixes(end+1, :) = [fx.t, t, nd.n_acc > n0];
        end
        queue(1, :) = [];  qz(1) = [];
    end
    if f.started
        X(k, :) = f.x';  Pd(k, :) = diag(f.P)';
        e = f.x - xt(k, :)';  e(3) = wrap_pi(e(3));
        nees(k) = e' * (f.P \ e);
    end
end
out.t = tr.t;  out.x = X;  out.Pdiag = Pd;  out.nees = nees;
err = X - xt;  err(:, 3) = wrap_pi(err(:, 3));  out.err = err;
out.nis = nd.nis;  out.nis_acc = nd.nis_acc;  out.fixes = fixes;
out.n_acc = nd.n_acc;  out.n_rej = nd.n_rej;  out.n_reanchor = nd.n_reanchor;
out.t_start = tr.t(find(~isnan(X(:, 1)), 1));
end
