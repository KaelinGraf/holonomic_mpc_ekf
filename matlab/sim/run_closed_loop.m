function L = run_closed_loop(prm, pose0, opts)
%RUN_CLOSED_LOOP Deployed design in closed loop on the truth plant (v2, 2026-10-16).
%   L = RUN_CLOSED_LOOP(prm, pose0, opts)
%   opts .use_ekf (true; false: true state fed back), .seed, .t_max (40 s of motion),
%        .noise_scale, .bias_scale (truth only; filter unaware), .log (false: summary +
%        per-tick log; true adds the per-IMU-sample log)
%   Timeline: armed at t = 0, robot at rest; accelerometer zeroed over [0, 2) s; the
%   first fix captured after that starts the filter; ACQUIRE -> APPROACH -> FINAL ->
%   DOCKED under sup_step. Plant at prm.sim.dt (1 ms): first-order wheel loops (tau),
%   a smooth slip offset (two cascaded 1 s low-pass filters on white noise), IMU truth
%   a_b = dv_b/dt + r S v_b. IMU at prm.imu.f; fixes captured at 15 Hz and delivered
%   prm.fid.delay later; controller at 20 Hz. The slip offset scales with the wheel-implied
%   speed (full above 0.05 m/s, none at rest), so a stopped robot does not creep.
if nargin < 3, opts = struct(); end
use_ekf = getfield_default(opts, 'use_ekf', true);
seed = getfield_default(opts, 'seed', 1);
t_max = getfield_default(opts, 't_max', 40);
ns = getfield_default(opts, 'noise_scale', 1);
bs = getfield_default(opts, 'bias_scale', 1);
full_log = getfield_default(opts, 'log', false);

dt = prm.sim.dt;  t_end = prm.sim.t_zero + 1 + t_max;  N = round(t_end/dt);
k_imu = false(N, 1);  k_imu(unique(max(1, round((prm.imu.T:prm.imu.T:t_end)/dt)))) = true;
k_fix = false(N, 1);  k_fix(unique(max(1, round((0:1/prm.fid.f:t_end)/dt)))) = true;
n_tick = round(prm.mpc.Ts/dt);
rs = RandStream('mt19937ar', 'Seed', seed);
imu_n = randn(rs, nnz(k_imu), 3);  fix_n = randn(rs, nnz(k_fix), 3);  slip_n = randn(rs, N, 2);
Jinv = prm.geom.Jinv;  Jfwd = pinv(Jinv);  tau = prm.plant.tau;  tsl = 1.0;
ss = prm.plant.slip_sigma;
% state
pose = pose0(:);  vb = zeros(3, 1);  w = zeros(4, 1);  w_cmd = zeros(4, 1);  sl1 = zeros(2, 1);  sl = zeros(2, 1);
f = ekf_create(prm);  nd = ekf_node_create(prm);
sv = struct('stage', 0, 'entered', 0, 'resume', 2, 'hold_ref', zeros(3, 1));
u_prev = zeros(3, 1);  misses = 0;  last_motion = -inf;  r_meas = 0;
queue = zeros(0, 5);  qz = {};  iI = 0;  iF = 0;  imu_times = [];
last_fix_true = -inf;
nT = ceil(N/n_tick);  T = nan(nT, 18);  it = 0;
if full_log, S = nan(nnz(k_imu), 15); end
t_motion = NaN;  t_dock = NaN;  t_abort = NaN;  pose_dock = nan(1, 3);  v_dock = NaN;
max_x = -inf;  v_contact = NaN;  n_fb = 0;  n_slack = 0;  x_prev = pose(1);
for k = 1:N
    t = k*dt;
    %% plant
    vb_old = vb;
    w = w + (dt/tau)*(w_cmd - w);
    sl1 = sl1 + (dt/tsl)*(-sl1) + sqrt(2)*ss*sqrt(2*dt/tsl)*slip_n(k, :)';
    sl = sl + (dt/tsl)*(sl1 - sl);
    sl = max(min(sl, 3*ss), -3*ss);
    vw = Jfwd*w;                                   % wheel-implied body velocity
    vb = vw + [sl*min(1, norm(vw(1:2))/0.05); 0];  % slip only while the wheels drive (none at rest)
    vm = 0.5*(vb_old + vb);  th = pose(3);
    pose(1:2) = pose(1:2) + dt*[cos(th), -sin(th); sin(th), cos(th)]*vm(1:2);
    pose(3) = wrap_pi(pose(3) + dt*vm(3));
    ab = (vb(1:2) - vb_old(1:2))/dt + vb(3)*[-vb(2); vb(1)];
    max_x = max(max_x, pose(1));
    if isnan(v_contact) && x_prev < 0 && pose(1) >= 0
        c = cos(pose(3)); s = sin(pose(3)); v_contact = c*vb(1) - s*vb(2);
    end
    x_prev = pose(1);
    %% IMU sample
    if k_imu(k)
        iI = iI + 1;  imu_times(end+1) = t; %#ok<AGROW>
        a = ab + prm.imu.bias_rest_a + (t >= prm.sim.t_zero)*bs*prm.imu.bias_a + ns*prm.imu.sigma_a*imu_n(iI, 1:2)';
        r_meas = vb(3) + bs*prm.imu.bias_r + ns*prm.imu.sigma_r*imu_n(iI, 3);
        if ~nd.zeroed
            if isempty(nd.zero_t0), nd.zero_t0 = t; end
            nd.zero_sum = nd.zero_sum + a;  nd.zero_n = nd.zero_n + 1;
            if t - nd.zero_t0 >= nd.t_zero, nd.offset = nd.zero_sum/nd.zero_n; nd.zeroed = true; end
        elseif use_ekf && f.started
            f = ekf_advance(f, t, [a - nd.offset; r_meas], t - last_motion >= prm.ekf.zv_quiet);
        end
        if full_log
            S(iI, :) = [t, pose', f.x', diag(f.P)', sv.stage];
        end
    end
    %% fix capture and delivery
    if k_fix(k)
        iF = iF + 1;
        fx = fiducial_measure(pose, prm, fix_n(iF, :)', ns);
        if fx.valid
            queue(end+1, :) = [t, t + prm.fid.delay, 0, 0, 0]; %#ok<AGROW>
            qz{end+1} = fx; %#ok<AGROW>
        end
    end
    while ~isempty(queue) && queue(1, 2) <= t + 1e-9
        fx = qz{1};  fx.t = queue(1, 1);
        if nd.zeroed && fx.t >= nd.zero_t0 + nd.t_zero
            last_fix_true = fx.t;
            if use_ekf, [f, nd] = ekf_node_fix(f, nd, fx, sv.stage == 1); end
        end
        queue(1, :) = [];  qz(1) = [];
    end
    %% controller tick
    if mod(k, n_tick) == 0
        if use_ekf
            if f.started, x = f.x; pd = diag(f.P); est_age = t - f.t_last; else, x = []; pd = inf(5, 1); est_age = inf; end
            fix_age = t - nd.last_accept;  rg = r_meas;
        else
            c = cos(pose(3)); s = sin(pose(3));
            x = [pose; c*vb(1) - s*vb(2); s*vb(1) + c*vb(2)];  pd = 1e-8*ones(5, 1);  est_age = 0;
            fix_age = t - last_fix_true;  rg = vb(3);
            if ~nd.zeroed, x = []; pd = inf(5, 1); est_age = inf; end
        end
        rate = sum(imu_times > t - 1)/prm.imu.f;
        in = struct('now', t, 'armed', true, 'x', x, 'p_diag', pd, 'fix_age', fix_age, ...
                    'est_age', est_age, 'imu_rate', rate, 'misses', misses, 'u_last', u_prev);
        st0 = sv.stage;
        [sv, out] = sup_step(sv, in, prm);
        if out.stage == 2 && isnan(t_motion), t_motion = t; end
        if out.stage == 4 && st0 ~= 4
            t_dock = t;  pose_dock = pose';  c = cos(pose(3)); s = sin(pose(3)); v_dock = c*vb(1) - s*vb(2);
        end
        if out.stage == 6 && st0 ~= 6, t_abort = t; end
        slack = NaN;  ef = NaN;  fb = false;  qp = NaN;
        if out.move
            tq = tic;
            [u_new, info] = mpc_step(x, rg, u_prev, out.yr, out.lim, prm, out.qw);
            qp = toc(tq);
            slack = info.slack;  ef = info.exitflag;  fb = info.fallback;
            misses = (misses + 1)*fb;  n_fb = n_fb + fb;  n_slack = n_slack + (slack > 1e-9);
        else
            u_new = zeros(3, 1);  misses = 0;
        end
        u_prev = u_new;
        if any(abs(u_prev) > prm.ekf.zv_cmd_eps), last_motion = t; end
        w_cmd = Jinv*u_prev;
        it = it + 1;
        if isempty(x), xe = nan(1, 5); else, xe = x'; end
        T(it, :) = [t, out.stage, pose', xe, u_prev', slack, ef, fb, qp, max(abs(w_cmd))];
        if (sv.stage == 4 && t > t_dock + 2) || (sv.stage == 6 && t > t_abort + 0.5) || ...
           (~isnan(t_motion) && t > t_motion + t_max)
            break
        end
    end
end
T = T(1:it, :);
L.tick = array2table(T, 'VariableNames', {'t', 'stage', 'x', 'y', 'th', 'xh', 'yh', 'thh', 'vxh', 'vyh', ...
                                          'u_b', 'v_b', 'r_b', 'slack', 'exitflag', 'fallback', 'qp_s', 'w_max_cmd'});
if full_log
    S = S(1:iI, :);
    L.imu = array2table(S, 'VariableNames', {'t', 'x', 'y', 'th', 'xh', 'yh', 'thh', 'vxh', 'vyh', ...
                                             'P11', 'P22', 'P33', 'P44', 'P55', 'stage'});
end
L.pose0 = pose0(:)';  L.seed = seed;  L.use_ekf = use_ekf;
L.docked = ~isnan(t_dock);  L.aborted = ~isnan(t_abort);
L.t_motion = t_motion;  L.t_dock = t_dock;  L.time_to_dock = t_dock - t_motion;
L.pose_at_dock = pose_dock;  L.v_at_dock = v_dock;
L.y_dock = pose_dock(2);  L.th_dock = pose_dock(3);
L.in_window = L.docked && abs(pose_dock(2)) <= prm.tol.y && abs(pose_dock(3)) <= prm.tol.th;
L.in_target = L.docked && abs(pose_dock(2)) <= prm.tgt.y && abs(pose_dock(3)) <= prm.tgt.th;
L.max_pen = max(0, max_x);  L.v_contact = v_contact;
L.n_fallback = n_fb;  L.n_slack_ticks = n_slack;
L.n_fix_acc = nd.n_acc;  L.n_fix_rej = nd.n_rej;  L.n_fix_refused = nd.n_refused;  L.n_reanchor = nd.n_reanchor;
L.nis = nd.nis;  L.final_stage = sv.stage;
L.max_w_cmd = max(T(:, 18));  L.max_abs_th = max(abs(T(:, 5)));
end
