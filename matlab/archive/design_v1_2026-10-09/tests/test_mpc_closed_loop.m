function r = test_mpc_closed_loop()
%TEST_MPC_CLOSED_LOOP Docking with perfect state feedback, three start poses, both variants.
%   The MPC drives the truth plant (wheel lag and slip included) with the true state fed
%   back, so this isolates the controller from the estimator. Pass, for every run:
%   docked within 40 s; lateral and yaw error at the moment of docking inside the
%   internal targets; penetration of the contact plane (true x) below the stopping
%   distance under the plant's wheel lag, v_final*tau, plus 5 mm (the kinematic MPC
%   cannot see the lag; the lag model of design doc Section 4.8 removes this overshoot);
%   closing speed at contact below 1.5 x the final speed limit; commanded wheel speeds
%   inside the limit; no solver fallbacks.
%   Catches: wrong frame in the command rotation (robot drives away), reference staging
%   that never switches, constraints that are assembled but not binding, a
%   no-penetration row that does not include the current state.
r.name = 'MPC closed loop, perfect state, 3 start poses x 2 variants';
starts = {[-2.0; 0.30; deg2rad(15)], [-1.5; -0.40; deg2rad(-20)], [-0.6; 0.02; deg2rad(2)]};
pass = true;  warn = false;  msg = '';
for variant = {'fullscale', 'demo'}
    prm = dock_params(variant{1});
    pen_allow = prm.mpc.contact_band + prm.lim.v_final*prm.plant.tau + 0.002;
    for i = 1:numel(starts)
        L = run_closed_loop(prm, starts{i}, struct('use_ekf', false, 'seed', 10 + i, 't_max', 40));
        docked = ~isnan(L.t_dock);
        if docked
            ey = abs(L.pose_at_dock(2));  eth = abs(wrap_pi(L.pose_at_dock(3)));
            ok = ey <= prm.tgt.y && eth <= prm.tgt.th && L.max_x < pen_allow ...
                 && abs(L.v_contact) <= 1.5*prm.lim.v_final ...
                 && L.max_th <= prm.mpc.lim.th_max + deg2rad(5) ...
                 && L.max_w_cmd <= prm.lim.w_max + 1e-3 && L.n_fallback == 0;
            warn = warn || L.max_x > 0.5*pen_allow;
            msg = [msg, sprintf('%s #%d: dock %.1f s, y %.1f mm, th %.2f deg, overshoot %.1f mm (allow %.0f), contact %.0f mm/s, max yaw %.0f deg; ', ...
                   variant{1}, i, L.t_dock, ey*1e3, rad2deg(eth), max(0, L.max_x)*1e3, pen_allow*1e3, L.v_contact*1e3, rad2deg(L.max_th))];
        else
            ok = false;
            msg = [msg, sprintf('%s #%d: NOT docked in 40 s (final x %.3f m, stage %d); ', variant{1}, i, L.pose(end,1), L.stage(end))];
        end
        pass = pass && ok;
    end
end
r.pass = pass;  r.warn = warn;  r.msg = msg;
end
