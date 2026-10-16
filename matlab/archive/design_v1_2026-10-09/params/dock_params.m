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

%% Timing
prm.imu.f     = 200;            % IMU rate [Hz]
prm.imu.T     = 1/prm.imu.f;    % IMU period [s]
prm.fid.f     = 15;             % fiducial pose solver rate [Hz]
prm.fid.delay = 0.050;          % capture-to-availability delay [s]
prm.gps.f     = 10;             % RTK-GPS rate [Hz]
prm.mpc.Ts    = 0.05;           % MPC period [s]

%% Variant-specific sensors, tolerances and geometry
switch prm.variant
    case 'fullscale'
        % IMU: dual-antenna GNSS/INS class (3DM-GQ7 / Ellipse-D), see design/imu_options_fullscale.md.
        % Per-sample white noise from VRW 0.012 m/s/sqrt(h) and ARW 0.15 deg/sqrt(h) at 100 Hz bandwidth.
        prm.imu.sigma_a = 0.002;            % accel white noise [m/s^2]
        prm.imu.sigma_r = deg2rad(0.03);    % gyro white noise [rad/s]
        prm.imu.bias_a  = [0.008; -0.005];  % residual accel bias in the truth model [m/s^2]: gravity leak at 0.05 deg roll/pitch error
        prm.imu.bias_r  = deg2rad(0.002);   % residual gyro bias in the truth model [rad/s]: GQ7 turn-on bias
        prm.fid.sigma_p  = 0.005;           % fiducial position noise at range <= d0 [m]
        prm.fid.sigma_th = deg2rad(0.5);    % fiducial yaw noise [rad]
        prm.fid.d0       = 1.0;             % range scale for noise growth [m]
        prm.gps.enabled  = true;            % RTK/INS present on the full-scale robot (navigation filter only)
        prm.gps.sigma        = 0.020;       % RTK position noise [m]
        prm.gps.sigma_psi    = deg2rad(0.3);% dual-antenna INS heading noise [rad] (GQ7: 0.25 deg at 1.5 m baseline)
        prm.gps.sigma_target = 0.020;       % uncertainty of the target's reported position [m]
        prm.gps.sigma_target_th = deg2rad(0.3); % uncertainty of the target's reported heading [rad]
        prm.tol.y  = 0.080;       prm.tgt.y  = 0.027;          % coupler lateral tolerance / internal target [m]
        prm.tol.th = deg2rad(20); prm.tgt.th = deg2rad(1.5);   % yaw tolerance / internal target [rad]
        prm.geom.Lk = 1.0;                  % chassis centre to coupler [m]
        prm.geom.Rw = 0.10;                 % wheel radius [m]              (placeholder)
        prm.geom.lx = 0.60;                 % half wheelbase [m]            (placeholder)
        prm.geom.ly = 0.45;                 % half track [m]                (placeholder)
        prm.lim.v_approach = 0.30;          % speed limits [m/s]
        prm.lim.v_final    = 0.05;
        prm.lim.a_max      = 0.30;          % acceleration limit, anti-slip [m/s^2]
        prm.lim.r_max      = 0.30;          % yaw-rate limit [rad/s]
        prm.lim.alpha_max  = 0.50;          % yaw acceleration limit [rad/s^2]
        prm.lim.w_max      = 4.0;           % wheel speed limit [rad/s]     (placeholder)
        prm.plant.tau      = 0.30;          % wheel-speed loop time constant in the truth plant [s]
        prm.plant.slip_sigma = 0.01;        % slowly varying body-velocity slip offset, std [m/s]
        prm.ekf.q_v  = 1.5e-6;              % per-step process-noise variance on velocity [(m/s)^2], tuned by sweep_ekf_tuning
        prm.ekf.q_th = 1.1e-10;             % per-step process-noise variance on heading [rad^2], tuned likewise
    case 'demo'
        % IMU: BNO085-class placeholder, to be measured on the bench. Assumes the robot
        % zeroes the linear-acceleration output while stationary at start-up, so only
        % the variation of the gravity leak with motion remains as a bias.
        prm.imu.sigma_a = 0.02;
        prm.imu.sigma_r = deg2rad(0.15);
        prm.imu.bias_a  = [0.03; -0.02];    % ~0.2 deg attitude error leaking gravity after zeroing
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
        prm.geom.Rw = 0.0508;               % VEX 4 in wheel [m]
        prm.geom.lx = 0.15;                 % half wheelbase [m]            (measure)
        prm.geom.ly = 0.15;                 % half track [m]                (measure)
        prm.lim.v_approach = 0.30;
        prm.lim.v_final    = 0.05;
        prm.lim.a_max      = 0.50;
        prm.lim.r_max      = 0.50;
        prm.lim.alpha_max  = 1.00;
        prm.lim.w_max      = 15.0;          % wheel speed limit [rad/s]     (measure)
        prm.plant.tau      = 0.10;          % wheel-speed loop time constant [s] (step test)
        prm.plant.slip_sigma = 0.01;
        prm.ekf.q_v  = 4.5e-6;              % per-step process-noise variance on velocity [(m/s)^2], tuned by sweep_ekf_tuning
        prm.ekf.q_th = 3.8e-9;              % per-step process-noise variance on heading [rad^2], tuned likewise
    otherwise
        error('dock_params:variant', 'Unknown variant "%s" (use ''fullscale'' or ''demo'')', variant);
end

%% EKF tuning (course Section 4.3: Q is the tuning knob, chosen with the consistency test)
% Input noise from the IMU, inflated by q_scale (2x sigma).
prm.ekf.q_scale = 4;
prm.ekf.Sigma_u = prm.ekf.q_scale * diag([prm.imu.sigma_a^2, prm.imu.sigma_a^2, prm.imu.sigma_r^2]);
% Process-noise floor: how much the filter distrusts its own velocity and heading
% prediction per step beyond the IMU white noise. The values q_v, q_th above are chosen
% with sweep_ekf_tuning so that the 20-run NEES sits near 5 with the marker visible.
% (Interpretation, if wanted: a per-step variance q gives sqrt(q*t/T) of uncertainty
% after t seconds; the full-scale q_v matches a 0.01 m/s^2 residual input bias over 3 s.)
prm.ekf.Qfloor = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);
% Initialisation: deployment form is the first fiducial fix with an unknown velocity;
% the tests use a draw about the truth (P0_handover) to exercise convergence.
prm.ekf.sigma_v0     = 0.1;             % [m/s] initial velocity uncertainty for the first-fix init
prm.ekf.P0_handover  = diag([0.05^2, 0.05^2, deg2rad(2)^2, 0.05^2, 0.05^2]);
prm.ekf.delay_comp   = true;            % ring-buffer delay compensation for the fiducial
prm.ekf.fid_gate_chi2 = 100;            % gross-error gate on the fiducial innovation (about 10 sigma): rejects a wrong
                                        % marker or a degenerate pose solve, never a plausible fix. A tight gate (e.g.
                                        % 99.9 %) makes the filter diverge when the target starts moving.
% The docking filter uses the IMU and the fiducial only. RTK position and INS heading
% (prm.gps) belong to the navigation filter; marker loss is handled by the supervisor.

%% MPC (design doc Section 4): kinematic model, incremental form, successive linearisation
prm.mpc.Np = 40;                        % prediction horizon [steps] (2 s at Ts = 0.05 s)
prm.mpc.Nc = 10;                        % control horizon [steps]
% Output weights by Bryson's rule: position on the coupler TOLERANCE, heading on the internal
% target (tgt.th = tgt.y/Lk). The asymmetry is deliberate: the heading column of A_k makes the
% heading a lateral actuator through the forward speed (a double-integrator path through the
% unmodelled wheel lag), and with the lateral error weighted on the internal target the
% optimiser steers with the heading: +-20 deg swings and a yaw-rate command alternating at the
% increment limit on the demonstrator. Nine times lighter on position, it crabs instead.
% Steady-state accuracy comes from the integral action, not the weight: the lateral error at
% docking stays inside the internal target in the closed-loop tests.
prm.mpc.Qy = diag(1 ./ [prm.tol.y, prm.tol.y, prm.tgt.th].^2);
% Increment weights by Bryson's rule on the largest allowed increment, times the scalar knob rw
prm.mpc.du_max = prm.mpc.Ts * [prm.lim.a_max; prm.lim.a_max; prm.lim.alpha_max];
prm.mpc.Ru = diag(1 ./ prm.mpc.du_max.^2);
prm.mpc.rw = 1;                         % the course's scalar tuning knob
% Limits per stage: the supervisor lowers lim.v_max when it switches to the final approach
prm.mpc.lim.du_max = prm.mpc.du_max;
prm.mpc.lim.v_max  = prm.lim.v_approach;  % translational cap |u_b| + |v_b| <= v_max. An L1 bound, not a box: a box is
                                          % 41 % faster along its diagonal and the optimiser yaws the body to use it
                                          % (23 deg on the demonstrator); the L1 set is the shape of the wheel-speed
                                          % limit at zero yaw rate, so the chassis is fastest along its own axes.
prm.mpc.lim.r_max  = prm.lim.r_max;       % yaw-rate cap |r_b| <= r_max
prm.mpc.lim.w_max  = prm.lim.w_max;
prm.mpc.contact_band = 0.010;           % [m] coupler engagement travel beyond first contact
prm.mpc.lim.x_max  = prm.mpc.contact_band;   % no-penetration: predicted x must not pass the end of the engagement travel
                                        % (a plane at first contact makes the controller bounce on it under wheel lag)
prm.mpc.lim.th_max = deg2rad(20);       % heading corridor: keeps the marker inside the camera's field of view
% Linear (exact) penalty on the output-constraint slack s (course 5.4.1). For the penalty to be
% exact it must exceed the largest constraint multiplier, which scales with the output weights:
% about 2*max(Qy)*Np*error. Scaling it with Qy keeps it exact for both variants.
prm.mpc.rho_pen    = 1e2 * prm.mpc.Np * max(diag(prm.mpc.Qy));
prm.mpc.v_max_final = prm.lim.v_final;  % translational cap in the final approach
% Reference staging (supervisor): standoff pose until aligned, then the origin
prm.mpc.standoff   = 0.30;              % [m] behind the contact plane
prm.mpc.align_tol  = [0.010; deg2rad(1.0)];   % lateral, yaw: switch to the final approach below these
prm.mpc.standoff_tol = 0.05;            % [m] longitudinal: close enough to the standoff pose to switch
prm.mpc.dock_tol   = 0.003;             % [m] estimated x within this of the contact plane: declare docked
prm.mpc.dock_vtol  = 0.02;              % [m/s] and slower than this
prm.mpc.dock_align = true;              % and the estimated lateral and heading errors inside the internal targets
                                        % (without it, a run can be declared docked while y is still converging:
                                        % 1 of 30 Monte Carlo runs on the demonstrator ended 0.7 mm outside the tolerance)
end
