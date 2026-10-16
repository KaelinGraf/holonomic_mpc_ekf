function out = run_ekf(sens, prm, tr, opts)
%RUN_EKF Multi-rate offline EKF loop over a simulated sensor record.
%   out = RUN_EKF(sens, prm, tr, opts)
%
%   Inputs
%     sens   from sim/simulate_sensors (help simulate_sensors for the field layout).
%            Only sens.imu and sens.fid are used; the docking filter has no RTK path.
%     prm    from params/dock_params
%     tr     truth trajectory from sim/make_truth_trajectory, or [] when unknown.
%            When given, out.err and out.nees are filled in.
%     opts   .delay_comp   use the buffered estimate at capture time for the fiducial
%                          innovation (default prm.ekf.delay_comp)
%            .init         'firstfix': x0 = [first fiducial fix; 0; 0],
%                                      P0 = blkdiag(R of that fix, prm.ekf.sigma_v0^2*eye(2))
%                                      (the deployment form, both variants)
%                          'handover' (default in the tests): x0 = truth(1) + draw from
%                                      prm.ekf.P0_handover after rng(opts.seed), P0 = P0_handover
%            .seed         rng seed for the handover draw (default 100)
%
%   Output fields (N = number of IMU samples)
%     out.t        [Nx1]  IMU time stamps
%     out.x        [Nx5]  estimate at each IMU sample, after all corrections due by then
%     out.Pdiag    [Nx5]  diag(P) at each IMU sample
%     out.fid_used [Nx1]  logical, a fiducial correction was applied at this sample
%     out.fid_rej  [Nx1]  logical, a fiducial fix was rejected by the gate at this sample
%     out.err      [Nx5]  x - truth with the heading component wrapped   (when tr given)
%     out.nees     [Nx1]  e'*inv(P)*e, e = x - truth wrapped               (when tr given)
%
%   Schedule (design/docking_formulation.pdf Section 3.6, ekf_equations.pdf step 10):
%     for each IMU sample k at time t:
%       predict with the IMU sample (skip at k = 1)
%       for each fiducial fix with t_avail <= t not yet processed:
%           correct with H = [eye(3) zeros(3,2)], R from the fix's sigmas,
%           x_ref = buffer(k_capture) when delay_comp, else the current x,
%           ang_idx = 3, gate = prm.ekf.fid_gate_chi2
%       store buffer(k) = x AFTER the corrections of this sample
%   The buffer is indexed by CAPTURE sample, not availability sample: the innovation
%   compares the fix with what the filter believed when the picture was taken.

if nargin < 3, tr = []; end
if nargin < 4, opts = struct(); end
delay_comp = getfield_default(opts, 'delay_comp', prm.ekf.delay_comp);
init_mode  = getfield_default(opts, 'init', 'handover');
seed       = getfield_default(opts, 'seed', 100);

T  = prm.imu.T;
N  = numel(sens.imu.t);
Hf = [eye(3), zeros(3,2)];              % fiducial pseudo-measurement: pose in T

%% Initialisation
switch init_mode
    case 'firstfix'     % deployment form: first fiducial fix, unknown velocity
        x = [sens.fid.z(1,:)'; 0; 0];
        P = blkdiag(diag([sens.fid.sigma_p(1)^2, sens.fid.sigma_p(1)^2, sens.fid.sigma_th(1)^2]), ...
                    prm.ekf.sigma_v0^2 * eye(2));
    case 'handover'     % test form: truth plus a draw from P0, exercises convergence
        if isempty(tr), error('run_ekf:init', 'handover init needs the truth trajectory'); end
        rng(seed);
        x = [tr.pose(1,:)'; tr.vel(1,1:2)'] + sqrt(diag(prm.ekf.P0_handover)) .* randn(5,1);
        x(3) = wrap_pi(x(3));
        P = prm.ekf.P0_handover;
    otherwise
        error('run_ekf:init', 'unknown init mode "%s"', init_mode);
end

%% Storage
X = zeros(N,5);  Pd = zeros(N,5);  buf = zeros(N,5);
fid_used = false(N,1);  fid_rej = false(N,1);  nees = nan(N,1);
have_truth = ~isempty(tr);
if have_truth, xt = [tr.pose, tr.vel(:,1:2)]; end

ip = 1;  nfid = numel(sens.fid.t_avail);

%% IMU-rate loop: predictor every sample, corrector on arrival
for k = 1:N
    t = sens.imu.t(k);
    if k > 1
        [x, P] = ekf_predict(x, P, [sens.imu.a(k,:)'; sens.imu.r(k)], T, prm.ekf.Sigma_u, prm.ekf.Qfloor);
    end

    % every fiducial fix that has become available by now, in arrival order
    while ip <= nfid && sens.fid.t_avail(ip) <= t + 1e-9
        if sens.fid.valid(ip)
            if delay_comp
                x_ref = buf(sens.fid.k_capture(ip), :)';
            else
                x_ref = x;
            end
            R = diag([sens.fid.sigma_p(ip)^2, sens.fid.sigma_p(ip)^2, sens.fid.sigma_th(ip)^2]);
            [x, P, ~, ~, acc] = ekf_update(x, P, sens.fid.z(ip,:)', Hf, R, x_ref, 3, prm.ekf.fid_gate_chi2);
            fid_used(k) = fid_used(k) || acc;
            fid_rej(k)  = fid_rej(k)  || ~acc;
        end
        ip = ip + 1;
    end

    % estimate at t_k after all corrections: delay buffer and logs
    buf(k,:) = x';
    X(k,:)   = x';
    Pd(k,:)  = diag(P)';
    if have_truth
        e = x - xt(k,:)';
        e(3) = wrap_pi(e(3));
        nees(k) = e' * (P \ e);
    end
end

%% Outputs
out.t        = sens.imu.t;
out.x        = X;
out.Pdiag    = Pd;
out.fid_used = fid_used;
out.fid_rej  = fid_rej;
out.opts     = struct('delay_comp', delay_comp, 'init', init_mode);
if have_truth
    err = X - xt;
    err(:,3) = wrap_pi(err(:,3));
    out.err  = err;
    out.nees = nees;
end
end
