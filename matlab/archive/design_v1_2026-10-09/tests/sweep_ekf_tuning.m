%SWEEP_EKF_TUNING Mean NEES and estimation error versus the velocity process noise q_v.
%   The course's tuning method (Section 4.3): sweep the process-noise entry, average 20
%   Monte Carlo runs per setting, and pick the value where the filter is consistent, i.e.
%   the mean NEES sits near 5 inside the printed band. Below the band the filter is
%   conservative (trusts the fixes more than it needs to), above it is overconfident
%   (trusts the IMU integration too much). The heading entry q_th is scaled with q_v.
%
%   Usage:  >> setup_paths; sweep_ekf_tuning      (or setup_paths('reference') first)

variants = {'fullscale', 'demo'};
factors  = [0.03, 0.1, 0.3, 1, 3, 10, 30];     % multiples of the value in dock_params
Nmc  = 20;
dropout = [18, 20];
[loN, hiN] = chi2_bounds(5*Nmc, 0.05);  loN = loN/Nmc;  hiN = hiN/Nmc;

for v = 1:numel(variants)
    prm0 = dock_params(variants{v});
    tr = make_truth_trajectory(prm0.imu.T);
    t  = tr.t;
    vis  = t > 2 & ~(t >= dropout(1) & t < dropout(2) + 0.5);   % converged, marker visible
    drop = t >= dropout(1) & t < dropout(2);
    fprintf('\n%s: %d-run mean NEES (consistent band %.2f to %.2f, ideal 5); q_v in dock_params = %.2e\n', ...
            variants{v}, Nmc, loN, hiN, prm0.ekf.q_v);
    fprintf('%10s %14s %14s %16s %16s\n', 'q_v', 'NEES visible', 'NEES dropout', 'pos RMS vis[mm]', 'vel RMS vis[mm/s]');
    for fac = factors
        prm = prm0;
        prm.ekf.q_v  = fac * prm0.ekf.q_v;
        prm.ekf.q_th = fac * prm0.ekf.q_th;
        prm.ekf.Qfloor = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);
        nees = zeros(numel(t), 1);  ep2 = 0;  ev2 = 0;
        for i = 1:Nmc
            s = simulate_sensors(tr, prm, 100 + i, struct('dropout', dropout));
            r = run_ekf(s, prm, tr, struct('seed', 200 + i));
            nees = nees + r.nees / Nmc;
            ep2 = ep2 + mean(hypot(r.err(vis,1), r.err(vis,2)).^2) / Nmc;
            ev2 = ev2 + mean(hypot(r.err(vis,4), r.err(vis,5)).^2) / Nmc;
        end
        fprintf('%10.2e %14.2f %14.2f %16.2f %16.2f\n', prm.ekf.q_v, mean(nees(vis)), mean(nees(drop)), sqrt(ep2)*1e3, sqrt(ev2)*1e3);
    end
end
