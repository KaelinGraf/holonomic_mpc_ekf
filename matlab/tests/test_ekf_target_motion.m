function r = test_ekf_target_motion()
%TEST_EKF_TARGET_MOTION The target starts moving at constant velocity during the approach.
%   The robot's IMU cannot see the target move, so the filter's velocity state must
%   re-converge to the new relative velocity purely from the fiducial fixes. A
%   constant target velocity is then absorbed exactly; the cost is a transient.
%   Two steps are applied at t = 6 s with a 0.5 s ramp: a creeping target (0.05 m/s)
%   and a moving one (0.20 m/s). Pass: the velocity error recovers to within 10 % of
%   the step inside 3 s in both cases, and after recovery the position error RMS
%   is no worse than 1.5x the undisturbed run. Peak position errors are reported.
%   Catches: a filter that cannot adjust its velocity from position fixes alone
%   (velocity process noise too small, or velocity treated as a pure IMU integral).
r.name = 'Target velocity step (constant-velocity target absorbed)';
prm = dock_params('fullscale');
t_step = 6;  t_ramp = 0.5;
tr0 = make_truth_trajectory(prm.imu.T);  t = tr0.t;
s0  = simulate_sensors(tr0, prm, 7);
o0  = run_ekf(s0, prm, tr0, struct('seed', 500));
after = t > t_step + 4 & t < 15.5;
ep0  = hypot(o0.err(:,1), o0.err(:,2));
ev0  = hypot(o0.err(:,4), o0.err(:,5));
rms0 = rms(ep0(after));
w    = round(0.5 / prm.imu.T);                    % 0.5 s moving average for the recovery check

pass = true;  msg = '';
for dv = [0.05, 0.20]
    dvec = [-dv; 0.3*dv];                         % mostly along -x, some lateral
    tr = apply_target_velocity_step(tr0, t_step, dvec, t_ramp);
    s  = simulate_sensors(tr, prm, 7);            % same noise seed as the baseline
    o  = run_ekf(s, prm, tr, struct('seed', 500));
    ev = hypot(o.err(:,4), o.err(:,5));
    ep = hypot(o.err(:,1), o.err(:,2));
    % recovered when the smoothed velocity error is back inside max(10 % of the step, 2x the noise floor)
    % (the search window is limited to 6 s after the step: with an open-loop truth the
    % relative range keeps growing afterwards, which raises the fiducial noise floor)
    thr = max(0.1*norm(dvec), 2*rms(ev0(after)));
    outside = find(t > t_step & t < t_step + 6 & movmean(ev, w) >= thr);
    if isempty(outside), t_rec = 0; else, t_rec = t(outside(end)) - t_step; end
    peak = max(ep(t >= t_step & t < t_step + 4));
    rms1 = rms(ep(after));
    ok = t_rec < 3 && rms1 < 1.5*rms0;
    pass = pass && ok;
    msg = [msg, sprintf('step %.2f m/s: recovery %.2f s, peak position error %.1f mm, RMS after %.1f mm (baseline %.1f mm). ', ...
                        dv, t_rec, peak*1e3, rms1*1e3, rms0*1e3)]; %#ok<AGROW>
end
r.pass = pass;  r.warn = false;  r.msg = msg;
end
