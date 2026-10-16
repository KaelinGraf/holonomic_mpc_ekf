function sens = simulate_sensors(tr, prm, seed, opts)
%SIMULATE_SENSORS IMU, fiducial and RTK-GPS records from a truth trajectory.
%   sens = SIMULATE_SENSORS(tr, prm, seed, opts)
%   opts.dropout = [t1 t2]  fiducial fixes captured in [t1, t2) are lost
%
%   IMU:      every sample; truth specific force + residual bias + white noise.
%   Fiducial: 15 Hz grid snapped to IMU samples; pose in T + range-dependent
%             noise; each fix becomes available prm.fid.delay after capture.
%   RTK/INS:  10 Hz; position and heading, already mapped into T through the
%             target's reported pose, whose error (position and heading) is a
%             constant offset for the run; plus white noise per fix. Generated for
%             the navigation studies; the docking filter does not use it.

if nargin < 3 || isempty(seed), seed = 1; end
if nargin < 4, opts = struct(); end
dropout = getfield_default(opts, 'dropout', []);
rng(seed);

T = prm.imu.T;
N = numel(tr.t);

sens.imu.t = tr.t;
sens.imu.a = tr.ab + prm.imu.bias_a(:)' + prm.imu.sigma_a*randn(N,2);
sens.imu.r = tr.r  + prm.imu.bias_r     + prm.imu.sigma_r*randn(N,1);

ic = unique(round((0:1/prm.fid.f:tr.t(end)) / T)) + 1;
ic = ic(ic <= N)';
n  = numel(ic);
d  = hypot(tr.pose(ic,1), tr.pose(ic,2));
sp = prm.fid.sigma_p * sqrt(1 + (d/prm.fid.d0).^2);
sens.fid.k_capture = ic;
sens.fid.t_capture = tr.t(ic);
sens.fid.t_avail   = tr.t(ic) + prm.fid.delay;
sens.fid.z = tr.pose(ic,:) + [sp.*randn(n,1), sp.*randn(n,1), prm.fid.sigma_th*randn(n,1)];
sens.fid.z(:,3) = wrap_pi(sens.fid.z(:,3));
sens.fid.sigma_p  = sp;
sens.fid.sigma_th = prm.fid.sigma_th*ones(n,1);
sens.fid.valid = true(n,1);
if ~isempty(dropout)
    sens.fid.valid(sens.fid.t_capture >= dropout(1) & sens.fid.t_capture < dropout(2)) = false;
end

if prm.gps.enabled
    ig = unique(round((0:1/prm.gps.f:tr.t(end)) / T)) + 1;
    ig = ig(ig <= N)';
    tgt_err = [prm.gps.sigma_target*randn(1,2), prm.gps.sigma_target_th*randn];
    n_g = numel(ig);
    sens.gps.k = ig;
    sens.gps.t = tr.t(ig);
    sens.gps.z = tr.pose(ig,:) + tgt_err + [prm.gps.sigma*randn(n_g,2), prm.gps.sigma_psi*randn(n_g,1)];
    sens.gps.z(:,3) = wrap_pi(sens.gps.z(:,3));
    sens.gps.R_rtk    = diag([prm.gps.sigma^2, prm.gps.sigma^2, prm.gps.sigma_psi^2]);
    sens.gps.R_target = diag([prm.gps.sigma_target^2, prm.gps.sigma_target^2, prm.gps.sigma_target_th^2]);
    sens.gps.target_err = tgt_err;
else
    sens.gps = [];
end
end
