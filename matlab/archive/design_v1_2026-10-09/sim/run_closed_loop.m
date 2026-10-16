function log = run_closed_loop(prm, pose0, opts)
%RUN_CLOSED_LOOP Closed-loop docking: truth plant, sensors, EKF, MPC and supervisor.
%   log = RUN_CLOSED_LOOP(prm, pose0, opts)
%   prm    from dock_params
%   pose0  [3x1] initial pose in the target frame (robot at rest)
%   opts   .use_ekf       true (default): estimate from the EKF functions on the path;
%                         false: perfect state feedback (test the MPC before the EKF exists)
%          .seed          noise seed (default 1)
%          .t_max         simulation limit [s] (default 40)
%          .rw            override prm.mpc.rw
%          .record_ticks  MPC tick numbers at which the predicted trajectory is stored
%          .unconstrained true: limits removed from the QP and the command clipped to the
%                         limits afterwards (the "with/without constraints" comparison)
%          .noise_scale   multiplies the simulated sensor noise only; the filter keeps its
%                         nominal R and Sigma_u (robustness to underestimated noise)
%          .bias_scale    multiplies the simulated IMU bias only
%
%   Timing: the plant, IMU and EKF step at the IMU period; the MPC runs every
%   round(Ts/T) samples and its wheel-speed command is held in between. Fiducial fixes
%   are captured on the 15 Hz grid and delivered prm.fid.delay later through a queue.
%
%   Supervisor stages (design doc Section 4.7):
%     1 approach: reference = standoff pose, approach speed limit
%     2 final:    reference = origin, final speed limit, entered when aligned at the standoff
%     3 docked:   zero command, entered when the estimated x is within dock_tol of the
%                 contact plane, the estimated speed is below dock_vtol and (dock_align)
%                 the estimated lateral and heading errors are inside the internal targets
%
%   Output log fields: t, pose (true), vb (true body velocity), xhat, u (held command),
%   w (held wheel command), stage, preds (struct array of stored predictions), and the
%   metrics t_dock, pose_at_dock, max_x (penetration), max_w_cmd, n_fallback, n_fid, n_rej.

if nargin < 3, opts = struct(); end
use_ekf = getfield_default(opts, 'use_ekf', true);
seed    = getfield_default(opts, 'seed', 1);
t_max   = getfield_default(opts, 't_max', 40);
rec     = getfield_default(opts, 'record_ticks', []);
unc     = getfield_default(opts, 'unconstrained', false);
ns      = getfield_default(opts, 'noise_scale', 1);
bs      = getfield_default(opts, 'bias_scale', 1);
if isfield(opts, 'rw') && ~isempty(opts.rw), prm.mpc.rw = opts.rw; end
rng(seed);

T  = prm.imu.T;  Ts = prm.mpc.Ts;  n_mpc = round(Ts/T);
N  = round(t_max/T) + 1;
t  = (0:N-1)' * T;
geom = prm.geom;
[~, Jinv_geom] = mecanum_ik(zeros(3,1), geom);

%% Fiducial capture schedule and delivery queue
k_cap = unique(round((0:1/prm.fid.f:t_max) / T)) + 1;
k_cap = k_cap(k_cap <= N);
n_delay = round(prm.fid.delay / T);
queue = [];                          % rows: [k_capture, k_avail, zx, zy, zth, sigma_p]

%% Plant, estimator and controller state
st = plant_init(pose0);
Hf = [eye(3), zeros(3,2)];
xhat = [pose0(:); 0; 0];             % first-fix style initialisation at t = 0
P    = blkdiag(diag([prm.fid.sigma_p^2, prm.fid.sigma_p^2, prm.fid.sigma_th^2]), prm.ekf.sigma_v0^2*eye(2));
buf  = zeros(N, 5);
u_prev = zeros(3,1);  w_cmd = zeros(4,1);  r_meas = 0;
stage = 1;  lim = prm.mpc.lim;  yr = [-prm.mpc.standoff; 0; 0];

%% Logs
L.pose = zeros(N,3);  L.vb = zeros(N,3);  L.xhat = zeros(N,5);  L.u = zeros(N,3);  L.w = zeros(N,4);
L.stage = zeros(N,1);  L.preds = struct('tick', {}, 't', {}, 'Y', {});  L.fallbacks = zeros(0, 11);
n_fallback = 0;  n_fid = 0;  n_rej = 0;  t_dock = NaN;  pose_at_dock = NaN(1,3);  tick = 0;

for k = 1:N
    %% Plant step (hold the wheel command) and IMU sample
    if k > 1
        [st, imu] = plant_step(st, w_cmd, prm, T);
        a_meas = imu.a + bs*prm.imu.bias_a(:) + ns*prm.imu.sigma_a*randn(2,1);
        r_meas = imu.r + bs*prm.imu.bias_r + ns*prm.imu.sigma_r*randn;
        if use_ekf
            [xhat, P] = ekf_predict(xhat, P, [a_meas; r_meas], T, prm.ekf.Sigma_u, prm.ekf.Qfloor);
        end
    end

    %% Fiducial: capture on the grid, deliver after the delay
    if any(k_cap == k)
        d  = hypot(st.pose(1), st.pose(2));
        sp = prm.fid.sigma_p*sqrt(1 + (d/prm.fid.d0)^2);
        z  = st.pose(:) + ns*[sp*randn(2,1); prm.fid.sigma_th*randn];
        z(3) = wrap_pi(z(3));
        queue(end+1, :) = [k, k + n_delay, z', sp]; %#ok<AGROW>
    end
    while ~isempty(queue) && queue(1, 2) <= k
        if use_ekf
            kc = queue(1, 1);  z = queue(1, 3:5)';  sp = queue(1, 6);
            R  = diag([sp^2, sp^2, prm.fid.sigma_th^2]);
            if kc < k, x_ref = buf(kc, :)'; else, x_ref = xhat; end     % zero delay: buffer not stored yet
            [xhat, P, ~, ~, acc] = ekf_update(xhat, P, z, Hf, R, x_ref, 3, prm.ekf.fid_gate_chi2);
            n_fid = n_fid + 1;  n_rej = n_rej + ~acc;
        end
        queue(1, :) = [];
    end
    if ~use_ekf
        Rth = rot2(st.pose(3));
        xhat = [st.pose(:); Rth*st.vb(1:2)];
        r_meas = st.vb(3);
    end
    buf(k, :) = xhat';

    %% Supervisor and MPC tick
    if mod(k-1, n_mpc) == 0 && stage < 3
        tick = tick + 1;
        if stage == 1 && abs(xhat(2)) < prm.mpc.align_tol(1) && abs(xhat(3)) < prm.mpc.align_tol(2) ...
                      && abs(xhat(1) + prm.mpc.standoff) < prm.mpc.standoff_tol
            stage = 2;  yr = zeros(3,1);
        end
        if stage == 2
            % lower the translational cap at the increment rate, never below what the current
            % command can reach in one tick: stepping it down would make the amplitude rows infeasible
            lim.v_max = max(prm.mpc.v_max_final, sum(max(abs(u_prev(1:2)) - prm.mpc.du_max(1:2), 0)));
        end
        aligned = ~getfield_default(prm.mpc, 'dock_align', false) || ...
                  (abs(xhat(2)) <= prm.tgt.y && abs(wrap_pi(xhat(3))) <= prm.tgt.th);
        if stage == 2 && xhat(1) >= -prm.mpc.dock_tol && hypot(xhat(4), xhat(5)) < prm.mpc.dock_vtol && aligned
            stage = 3;  t_dock = t(k);  pose_at_dock = st.pose(:)';
        end
        if stage < 3
            u_before = u_prev;
            if unc
                lim_free = lim;  lim_free.du_max = 1e3*[1;1;1];  lim_free.v_max = 1e3;  lim_free.r_max = 1e3;
                lim_free.w_max = 1e6;  lim_free.x_max = 1e3;  lim_free.th_max = 1e3;
                [u_prev, info] = mpc_step(xhat, r_meas, u_prev, yr, lim_free, prm);
                u_prev = saturate_cmd(u_prev, u_before, lim, Jinv_geom);
            else
                [u_prev, info] = mpc_step(xhat, r_meas, u_prev, yr, lim, prm);
            end
            n_fallback = n_fallback + info.fallback;
            if info.fallback
                L.fallbacks(end+1, :) = [tick, t(k), info.exitflag, stage, xhat(1), u_before', lim.v_max, lim.r_max, lim.w_max]; %#ok<AGROW>
            end
            w_cmd = mecanum_ik(u_prev, geom);
            if any(rec == tick)
                L.preds(end+1) = struct('tick', tick, 't', t(k), 'Y', info.Y); %#ok<AGROW>
            end
        else
            u_prev = zeros(3,1);  w_cmd = zeros(4,1);
        end
    end

    L.pose(k,:) = st.pose';  L.vb(k,:) = st.vb';  L.xhat(k,:) = xhat';
    L.u(k,:) = u_prev';  L.w(k,:) = w_cmd';  L.stage(k) = stage;
    if stage == 3 && t(k) > t_dock + 2, break; end        % two seconds after docking is enough
end

%% Trim and metrics
kk = 1:k;
log.t = t(kk);  log.pose = L.pose(kk,:);  log.vb = L.vb(kk,:);  log.xhat = L.xhat(kk,:);
log.u = L.u(kk,:);  log.w = L.w(kk,:);  log.stage = L.stage(kk);  log.preds = L.preds;
log.t_dock = t_dock;  log.pose_at_dock = pose_at_dock;
log.max_x = max(log.pose(:,1));
log.max_th = max(abs(log.pose(:,3)));
% closing speed at the first crossing of the contact plane (or at the end if never crossed)
kc = find(log.pose(:,1) >= 0, 1);  if isempty(kc), kc = numel(kk); end
log.v_contact = log.vb(kc, 1);
log.max_w_cmd = max(abs(log.w(:)));
log.n_fallback = n_fallback;  log.n_fid = n_fid;  log.n_rej = n_rej;
log.fallbacks = L.fallbacks;    % columns: tick, t, exitflag, stage, xhat_x, u_prev (3), v_max, r_max, w_max
log.use_ekf = use_ekf;  log.rw = prm.mpc.rw;
end

function u = saturate_cmd(u, u_prev, lim, Jinv)
%SATURATE_CMD What an unconstrained controller gets: the increment, L1 speed cap, yaw-rate
%   and wheel limits applied after the optimisation by clipping and scaling.
du = max(-lim.du_max(:), min(lim.du_max(:), u(:) - u_prev(:)));
u  = u_prev(:) + du;
s1 = abs(u(1)) + abs(u(2));
if s1 > lim.v_max, u(1:2) = u(1:2) * lim.v_max / s1; end
u(3) = max(-lim.r_max, min(lim.r_max, u(3)));
wm = max(abs(Jinv*u));
if wm > lim.w_max, u = u * lim.w_max / wm; end
end
