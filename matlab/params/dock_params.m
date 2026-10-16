function prm = dock_params(variant)
%DOCK_PARAMS Parameters of the DEPLOYED docking estimator, controller and supervisor (v2).
%   prm = DOCK_PARAMS('fullscale')  998 x 598 mm barrier-robot module (simulation only)
%   prm = DOCK_PARAMS('demo')       scale prototype (VEX chassis, Raspberry Pi 5)
%
%   v2 (2026-10-16) follows MATLAB_RERUN_BRIEF.md: the design as deployed and tuned on the
%   scale prototype (roll-back-and-replay EKF, N_p = 80, per-axis caps, seven-stage
%   supervisor), at both scales. The v1 design (2026-10-09) is archived in
%   matlab/archive/design_v1_2026-10-09.
%
%   Value sources, marked per line:
%     [deployed]  the prototype's deployed value as given in the brief (dock_demo.yaml)
%     [brief]     a value the brief states for both scales or for full scale
%     [x4]        full scale: the prototype's length scaled with the capture window (x4),
%                 the scaling rule of 2026-10-16; angles and times unchanged
%     [choice]    full scale: chosen here where the brief leaves it open (see results/rerun_v2/RESULTS.md)
%     [design]    unchanged from the v1 design (design doc / old parameter file)
%     [placeholder] demo geometry not available on this machine (dock_demo.yaml); stand-in

if nargin < 1, variant = 'fullscale'; end
prm.variant = lower(variant);
prm.version = 'v2-deployed-2026-10-16';

prm.fid.f      = 15;
prm.fid.delay  = 0.050;
prm.mpc.Ts     = 0.05;
prm.ekf.T0     = 0.005;
prm.sim.dt     = 0.001;
prm.sim.t_zero = 2.0;

switch prm.variant
    case 'fullscale'
        prm.imu.f       = 200;
        prm.imu.sigma_a = 0.002;
        prm.imu.sigma_r = deg2rad(0.03);
        prm.imu.bias_rest_a = [0.05; -0.03];
        prm.imu.bias_a  = [0.008; -0.005];
        prm.imu.bias_r  = deg2rad(0.002);
        prm.imu.mount_yaw = 0;
        prm.ekf.q_v  = 4.5e-8;
        prm.ekf.q_th = 3.3e-12;
        prm.fid.model    = 'law';
        prm.fid.square   = 0.040;
        prm.fid.f_px     = 580;
        prm.fid.z_scale  = 672/580;
        prm.fid.law_rot_a = [1.1196e-4, 1.5808e-4, 3.8022e-5];
        prm.fid.law_rot_k = [0.9674, 0.8074, 1.0171];
        prm.fid.law_tr_a  = [5.2027e-5, 4.6866e-5, 1.0236e-4];
        prm.fid.law_tr_k  = [1.5016, 1.4944, 1.7337];
        prm.fid.px_min   = 16;
        prm.fid.range_max = prm.fid.f_px*prm.fid.square/prm.fid.px_min;
        prm.fid.hfov     = deg2rad(28.9);
        prm.fid.vfov     = deg2rad(22.5);
        prm.geom.Rw = 0.1016;
        prm.geom.lx = 0.36;
        prm.geom.ly = 0.24;
        prm.geom.cam   = [0.45; 0];
        prm.geom.board = [0.65; 0];
        prm.geom.psi_n = pi;
        prm.geom.Lk = 0.65;
        prm.tol.y  = 0.048;   prm.tgt.y  = 0.016;
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.4);
        prm.lim.a_max     = 0.30;
        prm.lim.alpha_max = 0.50;
        prm.lim.r_max     = 0.30;
        prm.lim.w_max     = 4.0;
        prm.cap.v_approach  = 0.30;
        prm.cap.v_align     = 0.05;
        prm.cap.v_final     = 0.05;
        prm.cap.v_lat_final = 0.02;
        prm.plant.tau        = 0.30;
        prm.plant.slip_sigma = 0.01;
        prm.sup.standoff    = 0.32;
        prm.sup.handover_y  = 0.040;
        prm.sup.handover_x  = 0.060;
        prm.sup.dock_x      = 0.020;
        prm.sup.start_sd    = 0.120;
        prm.sup.abort_sd    = 0.200;
        prm.sup.lookat_d    = [1.20, 0.24];
        prm.sup.slow_d      = [1.20, 0.32];
        prm.sup.qw_d        = [3.20, 1.00];
        prm.ekf.reanchor_dist = 0.120;
        prm.ekf.zv_sigma_v    = 0.020;
        prm.ekf.zv_pos_rate   = 0.20^2;
    case 'demo'
        prm.imu.f       = 150;
        prm.imu.sigma_a = 0.022;
        prm.imu.sigma_r = deg2rad(0.026);
        prm.imu.bias_rest_a = [0.10; -0.05];
        prm.imu.bias_a  = [0.03; -0.02];
        prm.imu.bias_r  = deg2rad(0.05);
        prm.imu.mount_yaw = 0;
        prm.ekf.q_v  = 4.5e-6;
        prm.ekf.q_th = 3.8e-9;
        prm.fid.model    = 'diag';
        prm.fid.sigma_p0 = 0.004;
        prm.fid.d0       = 1.0;
        prm.fid.sigma_th = deg2rad(0.5);
        prm.fid.range_max = 2.5;
        prm.fid.hfov     = deg2rad(42);
        prm.fid.vfov     = deg2rad(22.5);
        prm.geom.Rw = 0.0275;  prm.geom.lx = 0.0765;  prm.geom.ly = 0.089;
        prm.geom.cam   = [0.135; 0];
        prm.geom.board = [0.2473; 0.0016];
        prm.geom.psi_n = deg2rad(178.82);
        prm.geom.Lk = 0.201;
        prm.tol.y  = 0.012;   prm.tgt.y  = 0.004;
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.5);
        prm.lim.a_max     = 0.50;
        prm.lim.alpha_max = 1.00;
        prm.lim.r_max     = 0.30;
        prm.lim.w_max     = 12.0;
        prm.cap.v_approach  = 0.18;
        prm.cap.v_align     = 0.04;
        prm.cap.v_final     = 0.09;
        prm.cap.v_lat_final = 0.02;
        prm.plant.tau        = 0.10;
        prm.plant.slip_sigma = 0.01;
        prm.sup.standoff    = 0.08;
        prm.sup.handover_y  = 0.010;
        prm.sup.handover_x  = 0.015;
        prm.sup.dock_x      = 0.005;
        prm.sup.start_sd    = 0.030;
        prm.sup.abort_sd    = 0.050;
        prm.sup.lookat_d    = [0.30, 0.06];
        prm.sup.slow_d      = [0.30, 0.08];
        prm.sup.qw_d        = [0.80, 0.25];
        prm.ekf.reanchor_dist = 0.030;
        prm.ekf.zv_sigma_v    = 0.005;
        prm.ekf.zv_pos_rate   = 0.05^2;
    otherwise
        error('dock_params:variant', 'Unknown variant "%s" (use ''fullscale'' or ''demo'')', variant);
end
prm.imu.T = 1/prm.imu.f;
prm.fid.sigma_th_floor = deg2rad(1.0);

prm.ekf.q_scale = 4;
prm.ekf.Sigma_u = prm.ekf.q_scale * diag([prm.imu.sigma_a^2, prm.imu.sigma_a^2, prm.imu.sigma_r^2]);
prm.ekf.Qfloor  = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);
prm.ekf.buf_len = round(2 * 0.5 * prm.imu.f);
prm.ekf.gate    = 100;
prm.ekf.sigma_v0 = 0.1;
prm.ekf.reanchor_n   = 5;
prm.ekf.reanchor_ang = deg2rad(3);
prm.ekf.zv_cmd_eps = 1e-4;
prm.ekf.zv_quiet   = 0.3;
prm.ekf.zv_th_rate = deg2rad(10)^2;

prm.mpc.Np = 80;
prm.mpc.Nc = 10;
prm.mpc.Qy = diag(1 ./ [prm.tol.y, prm.tgt.y, prm.tgt.th].^2);
prm.mpc.du_max = prm.mpc.Ts * [prm.lim.a_max; prm.lim.a_max; prm.lim.alpha_max];
prm.mpc.Ru = diag(1 ./ prm.mpc.du_max.^2);
prm.mpc.rw = [16; 16; 64];
prm.mpc.qw_approach_far  = [1; 0.25; 2];
prm.mpc.qw_approach_near = [1; 1; 2];
prm.mpc.qw_final         = [1; 1; 10];
prm.mpc.x_max  = 0.010;
prm.mpc.th_max = pi;
prm.mpc.rho_s  = 5.8e6;
prm.mpc.w_eps  = 200;
prm.mpc.diag_exact = false;

prm.sup.handover_th = deg2rad(2);
prm.sup.dock_v      = 0.020;
prm.sup.hold_after  = 0.5;
prm.sup.abort_after = 2.0;
prm.sup.stale       = 0.1;
prm.sup.imu_rate_min = 0.9;
prm.sup.miss_max    = 2;

L = prm.geom.lx + prm.geom.ly;
prm.geom.Jinv = (1/prm.geom.Rw) * [1, -1, -L; 1, 1, L; 1, 1, -L; 1, -1, L];
end
