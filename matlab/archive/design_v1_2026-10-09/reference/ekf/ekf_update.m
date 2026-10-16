function [x, P, nu, S, accepted] = ekf_update(x, P, z, H, R, x_ref, ang_idx, gate)
%EKF_UPDATE [REFERENCE] Measurement correction with delayed reference and chi-square gate.
%   Sealed reference implementation. Compare against it only after writing your own.
%   Interface: see matlab/ekf/ekf_update.m. Equations: docking_formulation.pdf (16)-(17).

if nargin < 6 || isempty(x_ref), x_ref = x; end
if nargin < 7, ang_idx = []; end
if nargin < 8 || isempty(gate), gate = inf; end

nu = z - H*x_ref;
nu(ang_idx) = wrap_pi(nu(ang_idx));
S = H*P*H' + R;
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
