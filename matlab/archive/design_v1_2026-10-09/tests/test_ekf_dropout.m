function r = test_ekf_dropout()
%TEST_EKF_DROPOUT Marker lost for 2 s during the final approach: coast on the IMU, stay honest.
%   With no marker the filter can only integrate the IMU, so the error grows with the
%   residual IMU bias. The test measures how long the position error stays inside the
%   coupler tolerance, which is the time the supervisor has to react before reverting
%   to navigation. Pass: at least 2 s at full scale (GQ7-class IMU) and at least three
%   camera frames (0.2 s) on the demonstrator (BNO085-class IMU), which is the time the
%   supervisor needs to declare the marker lost and stop; the error must stay inside its
%   3-sigma bound throughout the coast, and the first fixes after the gap must be accepted.
%   Catches: process noise too small to cover the coast (bound does not grow, the
%   returning fix is rejected or causes a jump), gate too tight, buffer errors after a gap.
r.name = 'Marker dropout, 2 s coast on the IMU (time inside tolerance)';
dropout = [18, 20];
t_req = struct('fullscale', 2.0, 'demo', 3/15);
pass = true;  warn = false;  msg = '';
for variant = {'fullscale', 'demo'}
    prm = dock_params(variant{1});
    tr  = make_truth_trajectory(prm.imu.T);
    t   = tr.t;
    win = t >= dropout(1) & t < dropout(2) + 0.1;
    tw  = t(win) - dropout(1);
    t_ok = inf;  worst_ratio = 0;  rej_after = 0;
    for i = 1:5
        s = simulate_sensors(tr, prm, 300 + i, struct('dropout', dropout));
        o = run_ekf(s, prm, tr, struct('seed', 400 + i));
        ep   = hypot(o.err(win, 1), o.err(win, 2));
        sig3 = 3*sqrt(o.Pdiag(win, 1) + o.Pdiag(win, 2));
        over = find(ep > prm.tol.y, 1);
        if isempty(over), t_ok = min(t_ok, diff(dropout)); else, t_ok = min(t_ok, tw(over)); end
        worst_ratio = max(worst_ratio, max(ep ./ sig3));
        rej_after = rej_after + nnz(o.fid_rej(t >= dropout(2) & t < dropout(2) + 0.5));
    end
    ok = t_ok >= t_req.(variant{1}) && worst_ratio < 1.2 && rej_after == 0;
    pass = pass && ok;
    warn = warn || worst_ratio > 1.0;
    msg = [msg, sprintf('%s: inside %.0f mm for %.2f s (need %.1f s), error/3sigma %.2f, fixes rejected on return %d. ', ...
                        variant{1}, prm.tol.y*1e3, t_ok, t_req.(variant{1}), worst_ratio, rej_after)];
end
r.pass = pass;  r.warn = warn;  r.msg = msg;
end
