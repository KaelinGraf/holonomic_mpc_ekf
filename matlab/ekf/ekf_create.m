function f = ekf_create(prm)
%EKF_CREATE Docking filter with a roll-back-and-replay buffer (deployed design, ekf.py).
%   f = EKF_CREATE(prm) returns an unstarted filter. Feed it with EKF_ADVANCE (IMU
%   samples) and EKF_APPLY_FIX (fixes, through EKF_NODE_FIX for the node-level rules).
%
%   Buffer: one entry per filter step: stamp, state, covariance, input, step length,
%   zero-velocity flag, and whether the state is valid (false before the first pose).
%   Length prm.ekf.buf_len entries (round(2 x 0.5 s x nominal IMU rate)). Entries are
%   recorded from the first IMU sample after the accelerometer zeroing, so the first pose
%   can be anchored at its capture step and replayed forward like any other fix.
L = prm.ekf.buf_len;
f.L        = L;
f.T0       = prm.ekf.T0;
f.buffer_s = 0.5;
f.Sigma_u  = prm.ekf.Sigma_u;
f.Qfloor   = prm.ekf.Qfloor;
f.gate     = prm.ekf.gate;
f.sigma_v0 = prm.ekf.sigma_v0;
f.zv_sigma_v  = prm.ekf.zv_sigma_v;
f.zv_pos_rate = prm.ekf.zv_pos_rate;
f.zv_th_rate  = prm.ekf.zv_th_rate;
f.started  = false;
f.x = nan(5, 1);
f.P = nan(5);
f.t_last     = NaN;
f.t_last_fix = -inf;
f.count = 0;
f.head  = 0;
f.stamp = nan(1, L);
f.X     = nan(5, L);
f.PP    = nan(5, 5, L);
f.U     = zeros(3, L);
f.Tk    = zeros(1, L);
f.ZV    = false(1, L);
f.valid = false(1, L);
f.n_replay = 0;
end
