function [f, nd] = ekf_node_fix(f, nd, fx, acquiring)
%EKF_NODE_FIX Node-level handling of one pose (ekf_node.py on_pose, resolve, reanchor).
%   fx: .t (capture stamp), .z, .R (first solution), optional .z2, .R2 (ambiguous pose).
%   nd: node state from ekf_node_create. Call only once the IMU is zeroed.
%   - resolve: with two solutions and the filter running, the one with the smaller NIS;
%     spread = diag(d^2) of their disagreement, added only where a fix is fused untested.
%   - first pose: initialise with R + spread.
%   - gated update; in ACQUIRE a fix that fails is fused ungated with R + spread.
%   - five gate rejections in a row, each within reanchor_dist and 3 deg of the one
%     before, re-anchor at the newest one's capture step (refusals for age do not count).
z = fx.z;  R = fx.R;  spread = zeros(3);
if isfield(fx, 'z2') && ~isempty(fx.z2)
    d = fx.z - fx.z2;  d(3) = wrap_pi(d(3));  spread = diag(d.^2);
    if f.started && ekf_nis(f, fx.z2, fx.R2, fx.t) < ekf_nis(f, fx.z, fx.R, fx.t)
        z = fx.z2;  R = fx.R2;
    end
end
Ru = R + spread;
if ~f.started
    f = ekf_fix(f, z, Ru, fx.t, 'init');
    nd.last_accept = fx.t;  nd.n_acc = nd.n_acc + 1;
    return
end
[f, acc, nis] = ekf_fix(f, z, R, fx.t, 'gated');
if acquiring && ~acc
    [f, acc, nis] = ekf_fix(f, z, Ru, fx.t, 'ungated');
end
nd.nis(end+1) = nis;  nd.nis_acc(end+1) = acc;
if acc
    nd.last_accept = fx.t;  nd.n_acc = nd.n_acc + 1;  nd.rej = zeros(3, 0);  nd.rej_t = [];
elseif ~isnan(nis)
    nd.n_rej = nd.n_rej + 1;
    nd.rej(:, end+1) = z;  nd.rej_t(end+1) = fx.t;  nd.rej_R{numel(nd.rej_t)} = Ru;
    if size(nd.rej, 2) > nd.n_anchor
        nd.rej = nd.rej(:, end-nd.n_anchor+1:end);  nd.rej_t = nd.rej_t(end-nd.n_anchor+1:end);
        nd.rej_R = nd.rej_R(end-nd.n_anchor+1:end);
    end
    if size(nd.rej, 2) == nd.n_anchor
        dz = diff(nd.rej, 1, 2);
        if all(abs(dz(1:2, :)) <= nd.anchor_p, 'all') && all(abs(wrap_pi(dz(3, :))) <= nd.anchor_th)
            f = ekf_fix(f, nd.rej(:, end), nd.rej_R{end}, nd.rej_t(end), 'anchor');
            nd.last_accept = nd.rej_t(end);  nd.n_reanchor = nd.n_reanchor + 1;
            nd.rej = zeros(3, 0);  nd.rej_t = [];  nd.rej_R = {};
        end
    end
else
    nd.n_refused = nd.n_refused + 1;
end
end

function nis = ekf_nis(f, z, R, stamp)
k = ekf_index_at(f, stamp);
if isempty(k), nis = inf; return; end
nu = z(:) - f.X(1:3, k);  nu(3) = wrap_pi(nu(3));
nis = nu' * ((f.PP(1:3, 1:3, k) + R) \ nu);
end
