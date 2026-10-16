function out = run_ekf(sens, prm, tr, opts)
%RUN_EKF [REFERENCE] Multi-rate offline EKF loop over a simulated sensor record.
%   Sealed reference implementation. Compare against it only after writing your own.
%   Interface and schedule: see matlab/ekf/run_ekf.m and docking_formulation.pdf 3.6.

if nargin < 3, tr = []; end
if nargin < 4, opts = struct(); end
delay_comp = getfield_default(opts, 'delay_comp', prm.ekf.delay_comp);
init_mode  = getfield_default(opts, 'init', 'handover');
seed       = getfield_default(opts, 'seed', 100);

T  = prm.imu.T;
N  = numel(sens.imu.t);
Hf = [eye(3), zeros(3,2)];

switch init_mode
    case 'firstfix'
        x = [sens.fid.z(1,:)'; 0; 0];
        P = blkdiag(diag([sens.fid.sigma_p(1)^2, sens.fid.sigma_p(1)^2, sens.fid.sigma_th(1)^2]), ...
                    prm.ekf.sigma_v0^2*eye(2));
    case 'handover'
        if isempty(tr), error('run_ekf:init', 'handover init needs the truth trajectory'); end
        rng(seed);
        x = [tr.pose(1,:)'; tr.vel(1,1:2)'] + sqrt(diag(prm.ekf.P0_handover)) .* randn(5,1);
        x(3) = wrap_pi(x(3));
        P = prm.ekf.P0_handover;
    otherwise
        error('run_ekf:init', 'unknown init mode "%s"', init_mode);
end

X   = zeros(N,5);  Pd = zeros(N,5);  buf = zeros(N,5);
fid_used = false(N,1);  fid_rej = false(N,1);
nees = nan(N,1);
have_truth = ~isempty(tr);
if have_truth, xt = [tr.pose, tr.vel(:,1:2)]; end

ip = 1;  nfid = numel(sens.fid.t_avail);

for k = 1:N
    t = sens.imu.t(k);
    if k > 1
        [x, P] = ekf_predict(x, P, [sens.imu.a(k,:)'; sens.imu.r(k)], T, prm.ekf.Sigma_u, prm.ekf.Qfloor);
    end

    while ip <= nfid && sens.fid.t_avail(ip) <= t + 1e-9
        if sens.fid.valid(ip)
            if delay_comp
                xref = buf(sens.fid.k_capture(ip), :)';
            else
                xref = x;
            end
            R = diag([sens.fid.sigma_p(ip)^2, sens.fid.sigma_p(ip)^2, sens.fid.sigma_th(ip)^2]);
            [x, P, ~, ~, acc] = ekf_update(x, P, sens.fid.z(ip,:)', Hf, R, xref, 3, prm.ekf.fid_gate_chi2);
            fid_used(k) = acc;
            fid_rej(k)  = ~acc;
        end
        ip = ip + 1;
    end

    buf(k,:) = x';
    X(k,:)   = x';
    Pd(k,:)  = diag(P)';
    if have_truth
        e = x - xt(k,:)';
        e(3) = wrap_pi(e(3));
        nees(k) = e' * (P \ e);
    end
end

out.t     = sens.imu.t;
out.x     = X;
out.Pdiag = Pd;
out.fid_used = fid_used;
out.fid_rej  = fid_rej;
out.opts  = struct('delay_comp', delay_comp, 'init', init_mode);
if have_truth
    err = X - xt;
    err(:,3) = wrap_pi(err(:,3));
    out.err  = err;
    out.nees = nees;
end
end
