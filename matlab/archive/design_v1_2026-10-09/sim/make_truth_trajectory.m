function tr = make_truth_trajectory(T, p0, segs)
%MAKE_TRUTH_TRAJECTORY Piecewise minimum-jerk docking approach in the target frame.
%   tr = MAKE_TRUTH_TRAJECTORY(T)            default staged approach (below)
%   tr = MAKE_TRUTH_TRAJECTORY(T, p0, segs)  p0 = [x; y; th] start pose,
%        segs(i) = struct('pf', [x; y; th], 'tf', move time [s], 'hold', dwell after the move [s])
%
%   Default: start (-2.0 m, 0.30 m, 15 deg) -> standoff (-0.30, 0, 0) in 14 s, hold 1.5 s,
%            -> docked (0, 0, 0) in 10 s, hold 2 s. Peak speeds 0.27 m/s and 0.056 m/s.
%
%   Each move is a quintic (minimum-jerk) blend, so velocity and acceleration are
%   analytic and the IMU truth follows directly. Output fields (N samples at period T):
%     t     [N x 1] time [s]
%     pose  [N x 3] x, y, theta in the target frame
%     vel   [N x 3] xdot, ydot, thetadot in the target frame
%     acc   [N x 3] xddot, yddot, thetaddot in the target frame
%     ab    [N x 2] body-frame specific force with gravity removed (what the IMU reports)
%     r     [N x 1] yaw rate (what the gyro reports)

if nargin < 2 || isempty(p0)
    p0 = [-2.0; 0.30; deg2rad(15)];
end
if nargin < 3 || isempty(segs)
    segs = [struct('pf', [-0.30; 0; 0], 'tf', 14, 'hold', 1.5), ...
            struct('pf', [0; 0; 0],     'tf', 10, 'hold', 2.0)];
end

t = 0;  pose = p0(:)';  vel = zeros(1,3);  acc = zeros(1,3);
tnow = 0;  pstart = p0(:);
for i = 1:numel(segs)
    tf = segs(i).tf;  pf = segs(i).pf(:);  d = (pf - pstart)';
    ts  = (T:T:tf + segs(i).hold)';            % start at T: the join sample already exists
    tau = min(ts/tf, 1);
    s   = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;
    sd  = (30*tau.^2 - 60*tau.^3 + 30*tau.^4) / tf;
    sdd = (60*tau - 180*tau.^2 + 120*tau.^3) / tf^2;
    t    = [t;    tnow + ts];           %#ok<AGROW>
    pose = [pose; pstart' + s*d];       %#ok<AGROW>
    vel  = [vel;  sd*d];                %#ok<AGROW>
    acc  = [acc;  sdd*d];               %#ok<AGROW>
    tnow = tnow + ts(end);
    pstart = pf;
end

th = pose(:,3);  c = cos(th);  sn = sin(th);
tr.T    = T;
tr.t    = t;
tr.pose = pose;
tr.vel  = vel;
tr.acc  = acc;
tr.ab   = [ c.*acc(:,1) + sn.*acc(:,2), -sn.*acc(:,1) + c.*acc(:,2)];   % R(th)' * [xdd; ydd]
tr.r    = vel(:,3);
end
