function [x, P, nu, S, accepted] = ekf_update(x, P, z, H, R, x_ref, ang_idx, gate)
%EKF_UPDATE Measurement correction with delayed reference and gross-error gate.
%   [x, P, nu, S, accepted] = EKF_UPDATE(x, P, z, H, R, x_ref, ang_idx, gate)
%
%   Inputs
%     x, P      current state and covariance
%     z, H, R   measurement, linear measurement matrix (p x 5), measurement covariance
%     x_ref     state used to form the innovation nu = z - H*x_ref. For a delayed
%               measurement pass the buffered estimate at capture time (Section 3.4);
%               the correction K*nu is still applied to the CURRENT x. Empty or
%               omitted means x itself.
%     ang_idx   indices of innovation components that are angles; wrapped with
%               utils/wrap_pi before use (omitted or empty: none)
%     gate      reject the update when nu'*inv(S)*nu > gate (omitted or empty: inf)
%
%   Outputs
%     x, P      corrected state and covariance (returned unchanged when rejected)
%     nu, S     innovation and innovation covariance
%     accepted  logical, false when the gate rejected the measurement
%
%   Equations: design/ekf_equations.pdf steps 7-9 (innovation, gain, Joseph form, gate).

if nargin < 6 || isempty(x_ref), x_ref = x; end
if nargin < 7, ang_idx = []; end
if nargin < 8 || isempty(gate), gate = inf; end

nu = z - H*x_ref;
nu(ang_idx) = wrap_pi(nu(ang_idx));
S  = H*P*H' + R;

accepted = (nu' * (S \ nu)) <= gate;
if ~accepted
    return
end

K = (P*H') / S;
x = x + K*nu;
x(3) = wrap_pi(x(3));
I = eye(numel(x));
P = (I - K*H)*P*(I - K*H)' + K*R*K';
P = (P + P')/2;
end
