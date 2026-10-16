function r = test_ekf_consistency()
%TEST_EKF_CONSISTENCY 20-run NEES with the marker visible, both variants.
%   The time-averaged NEES of a consistent 5-state filter sits near 5. Above the
%   95 % band the filter is overconfident (FAIL: it claims less error than it has,
%   which is dangerous for the MPC). Below the band it is conservative (WARN: safe,
%   but the tuning is leaving performance on the table). Also requires the error
%   to sit inside the 3-sigma bounds at least 95 % of the time.
%   Catches: Q or R mis-scaled, covariance not propagated, delay errors that bias
%   the estimate, wrong NEES (unwrapped heading error).
r.name = 'EKF consistency (20-run NEES, marker visible)';
Nmc = 20;  dropout = [18, 20];
[loN, hiN] = chi2_bounds(5*Nmc, 0.05);  loN = loN/Nmc;  hiN = hiN/Nmc;
pass = true;  warn = false;  msg = '';
for variant = {'fullscale', 'demo'}
    prm = dock_params(variant{1});
    tr  = make_truth_trajectory(prm.imu.T);
    t   = tr.t;
    vis = t > 2 & ~(t >= dropout(1) & t < dropout(2) + 0.5);
    nees = zeros(numel(t), 1);  in3 = 0;
    for i = 1:Nmc
        s = simulate_sensors(tr, prm, 100 + i, struct('dropout', dropout));
        o = run_ekf(s, prm, tr, struct('seed', 200 + i));
        nees = nees + o.nees / Nmc;
        in3  = in3 + mean(all(abs(o.err(vis, :)) <= 3*sqrt(o.Pdiag(vis, :)), 2)) / Nmc;
    end
    m = mean(nees(vis));
    pass = pass && m <= hiN && in3 >= 0.95;
    warn = warn || m < loN;
    msg = [msg, sprintf('%s: NEES %.2f (band %.2f-%.2f), inside 3 sigma %.1f%%. ', variant{1}, m, loN, hiN, in3*100)];
end
r.pass = pass;  r.warn = warn;  r.msg = msg;
end
