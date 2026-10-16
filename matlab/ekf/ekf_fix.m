function [f, accepted, nis] = ekf_fix(f, z, R, stamp, mode)
%EKF_FIX Apply a fix at its capture stamp by roll-back and replay (ekf.py update/reanchor/initialise).
%   mode 'gated'   innovation against the buffered state at the capture step, NIS <= gate,
%                  Joseph form, then replay every stored step after it (inputs, lengths,
%                  zero-velocity resets). Refused (accepted false, nis NaN) if the stamp
%                  predates the buffer or is older than the last fix applied.
%        'ungated' the same without the gate (ACQUIRE).
%        'init'    start the filter: x0 = [z; 0; 0], P0 = blkdiag(R, sigma_v0^2 I2) at stamp.
%        'anchor'  re-anchor: as init, but placed at the capture step and replayed.
accepted = false;  nis = NaN;
switch mode
    case 'init'
        f.x = [z(:); 0; 0];
        f.P = blkdiag(R, f.sigma_v0^2*eye(2));
        f.started = true;
        f = ekf_push(f, stamp, zeros(3,1), 0, false);
        f.t_last = stamp;  f.t_last_fix = stamp;  accepted = true;
        return
    case 'anchor'
        k = ekf_index_at(f, stamp);
        if isempty(k), [f, accepted, nis] = ekf_fix(f, z, R, stamp, 'init'); return; end
        f.x = [z(:); 0; 0];  f.x(3) = wrap_pi(f.x(3));
        f.P = blkdiag(R, f.sigma_v0^2*eye(2));
        f = replay_from(f, k);
        f.t_last_fix = stamp;  accepted = true;
        return
end
k = ekf_index_at(f, stamp);
if isempty(k) || stamp < f.t_last_fix, return; end
xk = f.X(:, k);  Pk = f.PP(:, :, k);
nu = z(:) - xk(1:3);  nu(3) = wrap_pi(nu(3));
S = Pk(1:3, 1:3) + R;
nis = nu' * (S \ nu);
if strcmp(mode, 'gated') && nis > f.gate, return; end
K = Pk(:, 1:3) / S;
x = xk + K*nu;  x(3) = wrap_pi(x(3));
IKH = eye(5);  IKH(:, 1:3) = IKH(:, 1:3) - K;
P = IKH*Pk*IKH' + K*R*K';
f.x = x;  f.P = 0.5*(P + P');
f = replay_from(f, k);
f.t_last_fix = stamp;  accepted = true;
end

function f = replay_from(f, k)
f.X(:, k) = f.x;  f.PP(:, :, k) = f.P;
while k ~= f.head
    k = mod(k, f.L) + 1;
    f = ekf_step(f, f.U(:, k), f.Tk(k), f.ZV(k));
    f.X(:, k) = f.x;  f.PP(:, :, k) = f.P;
    f.n_replay = f.n_replay + 1;
end
end
