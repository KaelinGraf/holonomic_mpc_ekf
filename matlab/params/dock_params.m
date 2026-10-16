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
%                 the author's rule of 2026-10-16; angles and times unchanged
%     [choice]    full scale: chosen here where the brief leaves it open (see the hand-back)
%     [design]    unchanged from the v1 design (design doc / old parameter file)
%     [placeholder] demo geometry not available on this machine (dock_demo.yaml); stand-in

if nargin < 1, variant = 'fullscale'; end
prm.variant = lower(variant);
prm.version = 'v2-deployed-2026-10-16';

%% Timing common to both scales
prm.fid.f      = 15;            % fiducial pose rate [Hz]                                    [brief]
prm.fid.delay  = 0.050;         % capture-to-availability delay [s], baseline                [brief]
prm.mpc.Ts     = 0.05;          % controller period [s]                                      [brief]
prm.ekf.T0     = 0.005;         % nominal filter step [s]; each IMU interval is split into
                                % n = max(1, round(dt/T0)) equal steps                     [brief]
prm.sim.dt     = 0.001;         % truth-plant integration step [s] (IMU samples, fixes and
                                % controller ticks fall on this grid)
prm.sim.t_zero = 2.0;           % accelerometer zeroing at rest after arming [s]             [brief]

switch prm.variant
    case 'fullscale'
        %% IMU (truth model)
        prm.imu.f       = 200;                  % nominal IMU rate [Hz]                     [brief]
        prm.imu.sigma_a = 0.002;                % per-sample white noise at rest [m/s^2]    [design]
        prm.imu.sigma_r = deg2rad(0.03);        % [rad/s]                                    [design]
        prm.imu.bias_rest_a = [0.05; -0.03];    % accelerometer bias present at arming: removed by the
                                                % 2 s zeroing (any constant value gives the same result)
        prm.imu.bias_a  = [0.008; -0.005];      % residual bias the zeroing cannot remove (attitude change
                                                % between rest and motion): the design's value [m/s^2]
        prm.imu.bias_r  = deg2rad(0.002);       % gyro bias, not zeroed [rad/s]              [design]
        prm.imu.mount_yaw = 0;                  % accelerometer mounting yaw [rad]           [brief]
        %% Filter tuning
        prm.ekf.q_v  = 4.5e-8;                  % process-noise floor per 5 ms step, velocity [(m/s)^2]
        prm.ekf.q_th = 3.3e-12;                 % heading [rad^2]. R1 sweep under the roll-back filter:
                                                % 0.03 x the design's 1.5e-6 / 1.1e-10 centres the 20-run
                                                % NEES (5.4); the design's values gave 1.03
        %% Fiducial: 200 mm ChArUco board (5 x 5, 40 mm squares), 4.0 mm lens at 580 px/rad (D1)
        prm.fid.model    = 'law';               % detector's range law (Appendix E)          [brief 2.3]
        prm.fid.square   = 0.040;               % [m]
        prm.fid.f_px     = 580;                 % [px/rad]
        prm.fid.z_scale  = 672/580;             % evaluate the law at z' = (672/580) l / square: pixel
                                                % size matched to the benchmark's median focal length
        prm.fid.law_rot_a = [1.1196e-4, 1.5808e-4, 3.8022e-5];   % rotation about x, y, z [rad]
        prm.fid.law_rot_k = [0.9674, 0.8074, 1.0171];
        prm.fid.law_tr_a  = [5.2027e-5, 4.6866e-5, 1.0236e-4];   % translation x, y, z [squares]
        prm.fid.law_tr_k  = [1.5016, 1.4944, 1.7337];
        prm.fid.px_min   = 16;                  % refuse squares under 16 px                 [brief]
        prm.fid.range_max = prm.fid.f_px*prm.fid.square/prm.fid.px_min;   % 1.45 m head-on
        prm.fid.hfov     = deg2rad(28.9);       % horizontal half-field [rad]                [brief]
        prm.fid.vfov     = deg2rad(22.5);       % vertical half-field (board at camera height here)
        %% Geometry
        prm.geom.Rw = 0.1016;                   % AndyMark 8 in mecanum [m]                  [brief]
        prm.geom.lx = 0.36;                     % half wheelbase [m]  [choice] <= 0.499 - Rw = 0.397
        prm.geom.ly = 0.24;                     % half track [m]      [choice] <= 0.299 - 0.040 (80 mm wheel)
        prm.geom.cam   = [0.45; 0];             % camera in the body frame [m]               [choice]
        prm.geom.board = [0.65; 0];             % board centre in the target frame [m]       [choice]
        prm.geom.psi_n = pi;                    % yaw of the board's outward normal in T: faces the robot
                                                % (docked camera-to-board range 0.20 m)     [brief 2.1]
        prm.geom.Lk = 0.65;                     % coupler lever arm consistent with theta_tgt = y_tgt/Lk
        %% Tolerances and internal targets
        prm.tol.y  = 0.048;   prm.tgt.y  = 0.016;                % [m]                         [brief]
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.4);     % [rad]                       [brief]
        %% Actuation limits
        prm.lim.a_max     = 0.30;               % [m/s^2]                                    [design]
        prm.lim.alpha_max = 0.50;               % [rad/s^2]                                  [design]
        prm.lim.r_max     = 0.30;               % [rad/s]                                    [brief]
        prm.lim.w_max     = 4.0;                % [rad/s] (0.41 m/s at the rim)              [brief]
        %% Speed caps
        prm.cap.v_approach  = 0.30;             % [m/s] Adam's docking speed                 [author]
        prm.cap.v_align     = 0.05;             % [m/s] cap at the standoff                  [choice]
        prm.cap.v_final     = 0.05;             % [m/s] Adam's final speed                   [author]
        prm.cap.v_lat_final = 0.02;             % [m/s] lateral cap in FINAL                 [choice]
        %% Truth plant
        prm.plant.tau        = 0.30;            % first-order wheel loops [s]                [design]
        prm.plant.slip_sigma = 0.01;            % slow body-velocity slip offset, std [m/s]  [design]
        %% Supervisor thresholds
        prm.sup.standoff    = 0.32;             % [m]                                        [x4]
        prm.sup.handover_y  = 0.040;            % |y| for APPROACH -> FINAL [m]              [x4]
        prm.sup.handover_x  = 0.060;            % within this of the standoff along x [m]    [x4]
        prm.sup.dock_x      = 0.020;            % docked when x >= -dock_x [m]               [x4]
        prm.sup.start_sd    = 0.120;            % ACQUIRE -> APPROACH: position std below [m] [x4]
        prm.sup.abort_sd    = 0.200;            % abort: position std above [m]              [x4]
        prm.sup.lookat_d    = [1.20, 0.24];     % look-at blends to theta = 0 over this distance to go [m] [x4]
        prm.sup.slow_d      = [1.20, 0.32];     % approach cap falls over this distance to go [m]          [x4]
        prm.sup.qw_d        = [3.20, 1.00];     % lateral stage weight ramps over this distance [m]         [x4]
        %% Re-anchoring and zero-velocity reset
        prm.ekf.reanchor_dist = 0.120;          % [m]                                        [x4]
        prm.ekf.zv_sigma_v    = 0.020;          % velocity std after a reset [m/s]           [x4]
        prm.ekf.zv_pos_rate   = 0.20^2;         % position variance growth [m^2 per s]       [x4]
    case 'demo'
        %% IMU (truth model): BNO085, measured at rest on the prototype
        prm.imu.f       = 150;                  % nominal IMU rate [Hz]                      [deployed]
        prm.imu.sigma_a = 0.022;                % [m/s^2]                                    [deployed]
        prm.imu.sigma_r = deg2rad(0.026);       % [rad/s]                                    [deployed]
        prm.imu.bias_rest_a = [0.10; -0.05];    % removed by the zeroing
        prm.imu.bias_a  = [0.03; -0.02];        % residual after zeroing                     [design]
        prm.imu.bias_r  = deg2rad(0.05);        % [rad/s]                                    [design]
        prm.imu.mount_yaw = 0;                  % the prototype's -2.81 deg hand-eye correction compensates
                                                % a physical misalignment the simulation does not have
        prm.ekf.q_v  = 4.5e-6;                  %                                            [deployed]
        prm.ekf.q_th = 3.8e-9;                  %                                            [deployed]
        %% Fiducial: the design's diagonal model (prototype board and lens not modelled here)
        prm.fid.model    = 'diag';
        prm.fid.sigma_p0 = 0.004;               % [m] at l <= d0                             [design]
        prm.fid.d0       = 1.0;                 % [m]
        prm.fid.sigma_th = deg2rad(0.5);        % [rad]                                      [design]
        prm.fid.range_max = 2.5;                % [m] (the prototype docked from 1.89 m)
        prm.fid.hfov     = deg2rad(42);         % [rad] the prototype's half field (dock_demo.yaml)
        prm.fid.vfov     = deg2rad(22.5);
        %% Geometry
        prm.geom.Rw = 0.0275;  prm.geom.lx = 0.0765;  prm.geom.ly = 0.089;   % measured [deployed]
        prm.geom.cam   = [0.135; 0];            % [m]                                        [deployed]
        prm.geom.board = [0.2473; 0.0016];      % [m]                                        [deployed]
        prm.geom.psi_n = deg2rad(178.82);       %                                            [deployed]
        prm.geom.Lk = 0.201;                    % built robot                                [deployed]
        %% Tolerances and internal targets
        prm.tol.y  = 0.012;   prm.tgt.y  = 0.004;                %                             [deployed]
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.5);     %                             [deployed]
        %% Actuation limits
        prm.lim.a_max     = 0.50;               %                                            [deployed]
        prm.lim.alpha_max = 1.00;               %                                            [deployed]
        prm.lim.r_max     = 0.30;               %                                            [brief]
        prm.lim.w_max     = 12.0;               %                                            [brief]
        %% Speed caps
        prm.cap.v_approach  = 0.18;             %                                            [deployed]
        prm.cap.v_align     = 0.04;
        prm.cap.v_final     = 0.09;
        prm.cap.v_lat_final = 0.02;
        %% Truth plant
        prm.plant.tau        = 0.10;            %                                            [design]
        prm.plant.slip_sigma = 0.01;
        %% Supervisor thresholds                                                             [deployed]
        prm.sup.standoff    = 0.08;
        prm.sup.handover_y  = 0.010;
        prm.sup.handover_x  = 0.015;
        prm.sup.dock_x      = 0.005;
        prm.sup.start_sd    = 0.030;
        prm.sup.abort_sd    = 0.050;
        prm.sup.lookat_d    = [0.30, 0.06];
        prm.sup.slow_d      = [0.30, 0.08];
        prm.sup.qw_d        = [0.80, 0.25];
        %% Re-anchoring and zero-velocity reset                                              [deployed]
        prm.ekf.reanchor_dist = 0.030;
        prm.ekf.zv_sigma_v    = 0.005;
        prm.ekf.zv_pos_rate   = 0.05^2;
    otherwise
        error('dock_params:variant', 'Unknown variant "%s" (use ''fullscale'' or ''demo'')', variant);
end
prm.imu.T = 1/prm.imu.f;
prm.fid.sigma_th_floor = deg2rad(1.0);  % filter's floor on the heading std of R_z            [brief]

%% Estimator (both scales)
prm.ekf.q_scale = 4;                    % filter input noise = 4 x the at-rest variance: per-sample std
                                        % full scale 0.004 m/s^2, 0.06 deg/s; prototype 0.044, 0.052
prm.ekf.Sigma_u = prm.ekf.q_scale * diag([prm.imu.sigma_a^2, prm.imu.sigma_a^2, prm.imu.sigma_r^2]);
prm.ekf.Qfloor  = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);   % per T0 step
prm.ekf.buf_len = round(2 * 0.5 * prm.imu.f);   % 200 entries full scale, 150 prototype       [brief]
prm.ekf.gate    = 100;                  % NIS gate on every gated fix                         [brief]
prm.ekf.sigma_v0 = 0.1;                 % velocity std at start and re-anchor [m/s]           [brief]
prm.ekf.reanchor_n   = 5;               % consecutive gate rejections                         [brief]
prm.ekf.reanchor_ang = deg2rad(3);      % each within reanchor_dist and 3 deg of the previous [brief]
prm.ekf.zv_cmd_eps = 1e-4;              % every command component within this of zero ...     [brief]
prm.ekf.zv_quiet   = 0.3;               % ... for this long [s]                               [brief]
prm.ekf.zv_th_rate = deg2rad(10)^2;     % heading variance growth during a reset [rad^2 per s][brief]

%% Controller (both scales)
prm.mpc.Np = 80;                        % 4 s                                                 [brief]
prm.mpc.Nc = 10;                        %                                                     [brief]
% Bryson in the deployed form: lateral weight from the internal target, x from the window
prm.mpc.Qy = diag(1 ./ [prm.tol.y, prm.tgt.y, prm.tgt.th].^2);    % full 434/3906/1675, demo 6944/62500/1459
prm.mpc.du_max = prm.mpc.Ts * [prm.lim.a_max; prm.lim.a_max; prm.lim.alpha_max];
prm.mpc.Ru = diag(1 ./ prm.mpc.du_max.^2);                         % full 4444/4444/1600, demo 1600/1600/400
prm.mpc.rw = [16; 16; 64];              % effort multipliers on the increments                [brief]
prm.mpc.qw_approach_far  = [1; 0.25; 2];% stage multipliers on Qy, APPROACH far ...           [brief]
prm.mpc.qw_approach_near = [1; 1; 2];   % ... ramping (smoothstep) to this over sup.qw_d
prm.mpc.qw_final         = [1; 1; 10];  % FINAL; HOLD keeps the stage it left                 [brief]
prm.mpc.x_max  = 0.010;                 % no-penetration limit: end of the engagement band [m][brief]
prm.mpc.th_max = pi;                    % heading corridor OFF (180 deg): heading in the cost only
prm.mpc.rho_s  = 5.8e6;                 % linear penalty on the shared slack                  [brief]
prm.mpc.w_eps  = 200;                   % quadratic slack weight (Hessian entry)              [brief]
prm.mpc.diag_exact = false;             % R8: also solve the hard programme every tick

%% Supervisor (both scales)
prm.sup.handover_th = deg2rad(2);       % |theta| for APPROACH -> FINAL                       [brief]
prm.sup.dock_v      = 0.020;            % docked when the estimated speed is below [m/s]      [brief]
prm.sup.hold_after  = 0.5;              % HOLD after this long without an accepted fix [s]    [brief]
prm.sup.abort_after = 2.0;              % ABORT after this long in HOLD [s]                   [brief]
prm.sup.stale       = 0.1;              % abort when the estimate is older than this [s]      [brief]
prm.sup.imu_rate_min = 0.9;             % ACQUIRE -> APPROACH needs the IMU at >= 90 % of nominal
prm.sup.miss_max    = 2;                % consecutive solver misses that trigger HOLD         [brief]

%% Wheel kinematics used by the controller and the plant
L = prm.geom.lx + prm.geom.ly;
prm.geom.Jinv = (1/prm.geom.Rw) * [1, -1, -L; 1, 1, L; 1, 1, -L; 1, -1, L];
end
