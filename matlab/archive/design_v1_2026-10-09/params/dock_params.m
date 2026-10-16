function prm = dock_params(variant)
%DOCK_PARAMS Parameter set for the docking estimator and controller study.
%   prm = DOCK_PARAMS('fullscale')  600 kg barrier robot (theory model)
%   prm = DOCK_PARAMS('demo')       tabletop VEX demonstrator
%
%   Noise values are per-sample standard deviations at the stated rates.
%   Everything here is a placeholder from design/docking_formulation.pdf until
%   replaced by datasheet or bench-measured values (see the parameter sheet).

if nargin < 1, variant = 'fullscale'; end
prm.variant = lower(variant);

prm.imu.f     = 200;
prm.imu.T     = 1/prm.imu.f;
prm.fid.f     = 15;
prm.fid.delay = 0.050;
prm.gps.f     = 10;
prm.mpc.Ts    = 0.05;

switch prm.variant
    case 'fullscale'
        prm.imu.sigma_a = 0.002;
        prm.imu.sigma_r = deg2rad(0.03);
        prm.imu.bias_a  = [0.008; -0.005];
        prm.imu.bias_r  = deg2rad(0.002);
        prm.fid.sigma_p  = 0.005;
        prm.fid.sigma_th = deg2rad(0.5);
        prm.fid.d0       = 1.0;
        prm.gps.enabled  = true;
        prm.gps.sigma        = 0.020;
        prm.gps.sigma_psi    = deg2rad(0.3);
        prm.gps.sigma_target = 0.020;
        prm.gps.sigma_target_th = deg2rad(0.3);
        prm.tol.y  = 0.080;       prm.tgt.y  = 0.027;
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.5);
        prm.geom.Lk = 1.0;
        prm.geom.Rw = 0.10;
        prm.geom.lx = 0.60;
        prm.geom.ly = 0.45;
        prm.lim.v_approach = 0.30;
        prm.lim.v_final    = 0.05;
        prm.lim.a_max      = 0.30;
        prm.lim.r_max      = 0.30;
        prm.lim.alpha_max  = 0.50;
        prm.lim.w_max      = 4.0;
        prm.plant.tau      = 0.30;
        prm.plant.slip_sigma = 0.01;
        prm.ekf.q_v  = 1.5e-6;
        prm.ekf.q_th = 1.1e-10;
    case 'demo'
        prm.imu.sigma_a = 0.02;
        prm.imu.sigma_r = deg2rad(0.15);
        prm.imu.bias_a  = [0.03; -0.02];
        prm.imu.bias_r  = deg2rad(0.05);
        prm.fid.sigma_p  = 0.004;
        prm.fid.sigma_th = deg2rad(0.5);
        prm.fid.d0       = 1.0;
        prm.gps.enabled  = false;
        prm.gps.sigma        = NaN;
        prm.gps.sigma_psi    = NaN;
        prm.gps.sigma_target = NaN;
        prm.gps.sigma_target_th = NaN;
        prm.tol.y  = 0.012;       prm.tgt.y  = 0.004;
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.5);
        prm.geom.Lk = 0.15;
        prm.geom.Rw = 0.0508;
        prm.geom.lx = 0.15;
        prm.geom.ly = 0.15;
        prm.lim.v_approach = 0.30;
        prm.lim.v_final    = 0.05;
        prm.lim.a_max      = 0.50;
        prm.lim.r_max      = 0.50;
        prm.lim.alpha_max  = 1.00;
        prm.lim.w_max      = 15.0;
        prm.plant.tau      = 0.10;
        prm.plant.slip_sigma = 0.01;
        prm.ekf.q_v  = 4.5e-6;
        prm.ekf.q_th = 3.8e-9;
    otherwise
        error('dock_params:variant', 'Unknown variant "%s" (use ''fullscale'' or ''demo'')', variant);
end

prm.ekf.q_scale = 4;
prm.ekf.Sigma_u = prm.ekf.q_scale * diag([prm.imu.sigma_a^2, prm.imu.sigma_a^2, prm.imu.sigma_r^2]);
prm.ekf.Qfloor = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);
prm.ekf.sigma_v0     = 0.1;
prm.ekf.P0_handover  = diag([0.05^2, 0.05^2, deg2rad(2)^2, 0.05^2, 0.05^2]);
prm.ekf.delay_comp   = true;
prm.ekf.fid_gate_chi2 = 100;

prm.mpc.Np = 40;
prm.mpc.Nc = 10;
prm.mpc.Qy = diag(1 ./ [prm.tol.y, prm.tol.y, prm.tgt.th].^2);
prm.mpc.du_max = prm.mpc.Ts * [prm.lim.a_max; prm.lim.a_max; prm.lim.alpha_max];
prm.mpc.Ru = diag(1 ./ prm.mpc.du_max.^2);
prm.mpc.rw = 1;
prm.mpc.lim.du_max = prm.mpc.du_max;
prm.mpc.lim.v_max  = prm.lim.v_approach;
prm.mpc.lim.r_max  = prm.lim.r_max;
prm.mpc.lim.w_max  = prm.lim.w_max;
prm.mpc.contact_band = 0.010;
prm.mpc.lim.x_max  = prm.mpc.contact_band;
prm.mpc.lim.th_max = deg2rad(20);
prm.mpc.rho_pen    = 1e2 * prm.mpc.Np * max(diag(prm.mpc.Qy));
prm.mpc.v_max_final = prm.lim.v_final;
prm.mpc.standoff   = 0.30;
prm.mpc.align_tol  = [0.010; deg2rad(1.0)];
prm.mpc.standoff_tol = 0.05;
prm.mpc.dock_tol   = 0.003;
prm.mpc.dock_vtol  = 0.02;
prm.mpc.dock_align = true;
end
