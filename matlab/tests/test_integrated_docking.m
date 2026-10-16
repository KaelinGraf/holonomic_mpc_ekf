function r = test_integrated_docking()
%TEST_INTEGRATED_DOCKING EKF and MPC together on the truth plant, both variants.
%   The full loop: IMU and delayed fiducial through the EKF on the path, MPC on the
%   estimate, supervisor staging. Pass per variant: docked within 40 s, lateral and
%   yaw error at docking inside the internal targets, penetration below the lag
%   stopping distance v_final*tau plus 5 mm, no solver fallbacks, no fiducial fixes
%   rejected by the gate.
%   Catches: estimator and controller frame or timing mismatches that the separate
%   tests cannot see, e.g. the MPC reading velocity in the wrong frame from the EKF.
r.name = 'Integrated docking, EKF in the loop, both variants';
pose0 = [-2.0; 0.30; deg2rad(15)];
pass = true;  warn = false;  msg = '';
for variant = {'fullscale', 'demo'}
    prm = dock_params(variant{1});
    pen_allow = prm.mpc.contact_band + prm.lim.v_final*prm.plant.tau + 0.002;   % engagement travel + lag coast
    L = run_closed_loop(prm, pose0, struct('use_ekf', true, 'seed', 21, 't_max', 40));
    docked = ~isnan(L.t_dock);
    if docked
        ey = abs(L.pose_at_dock(2));  eth = abs(wrap_pi(L.pose_at_dock(3)));
        ok = ey <= prm.tgt.y && eth <= prm.tgt.th && L.max_x < pen_allow && L.n_fallback == 0 && L.n_rej == 0 ...
             && L.max_th <= prm.mpc.lim.th_max + deg2rad(5);
        warn = warn || L.max_x > 0.5*pen_allow;
        msg = [msg, sprintf('%s: dock %.1f s, y %.1f mm, th %.2f deg, overshoot %.1f mm (allow %.0f), max yaw %.0f deg, fixes %d, rejected %d, fallbacks %d; ', ...
               variant{1}, L.t_dock, ey*1e3, rad2deg(eth), max(0, L.max_x)*1e3, pen_allow*1e3, rad2deg(L.max_th), L.n_fid, L.n_rej, L.n_fallback)]; %#ok<AGROW>
    else
        ok = false;
        msg = [msg, sprintf('%s: NOT docked in 40 s (final x %.3f m, stage %d); ', variant{1}, L.pose(end,1), L.stage(end))]; %#ok<AGROW>
    end
    pass = pass && ok;
end
r.pass = pass;  r.warn = warn;  r.msg = msg;
end
