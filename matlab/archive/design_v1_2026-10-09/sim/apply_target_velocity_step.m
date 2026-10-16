function tr = apply_target_velocity_step(tr, t_step, dv, t_ramp)
%APPLY_TARGET_VELOCITY_STEP Make the target start translating at constant velocity.
%   tr = APPLY_TARGET_VELOCITY_STEP(tr, t_step, dv, t_ramp)
%   dv      [2x1] target velocity in the target frame reached after the ramp [m/s]
%   t_step  ramp start [s];  t_ramp  ramp duration [s] (default 0.5)
%
%   The fixture trajectory is the robot's motion RELATIVE to the target. A target
%   that moves at velocity dv shifts the relative position by -dv*(t - t_step) and
%   the relative velocity by -dv, but leaves the robot's own IMU untouched: tr.ab
%   and tr.r stay as they are, because the target's motion is invisible to the
%   robot's accelerometer and gyro. tr.acc (relative) is updated for completeness.
%   Rotation of the target is not modelled here (see docking_formulation.pdf, the
%   moving-target section).

if nargin < 4 || isempty(t_ramp), t_ramp = 0.5; end
dv  = dv(:)';
tau = tr.t - t_step;
s   = min(max(tau / t_ramp, 0), 1);
sd  = (tau >= 0 & tau < t_ramp) / t_ramp;
int_s = zeros(size(tau));
in_ramp = tau >= 0 & tau < t_ramp;
after   = tau >= t_ramp;
int_s(in_ramp) = tau(in_ramp).^2 / (2*t_ramp);
int_s(after)   = tau(after) - t_ramp/2;
tr.pose(:,1:2) = tr.pose(:,1:2) - int_s * dv;
tr.vel(:,1:2)  = tr.vel(:,1:2)  - s * dv;
tr.acc(:,1:2)  = tr.acc(:,1:2)  - sd * dv;
end
