function nd = ekf_node_create(prm)
%EKF_NODE_CREATE Node-level state for ekf_node_fix (zeroing, fix bookkeeping, re-anchoring).
nd.last_accept = -inf;  nd.n_acc = 0;  nd.n_rej = 0;  nd.n_refused = 0;  nd.n_reanchor = 0;
nd.nis = [];  nd.nis_acc = false(1, 0);
nd.rej = zeros(3, 0);  nd.rej_t = [];  nd.rej_R = {};
nd.n_anchor  = prm.ekf.reanchor_n;
nd.anchor_p  = prm.ekf.reanchor_dist;
nd.anchor_th = prm.ekf.reanchor_ang;
nd.zeroed = false;  nd.zero_t0 = [];  nd.zero_sum = zeros(2, 1);  nd.zero_n = 0;  nd.offset = zeros(2, 1);
nd.t_zero = prm.sim.t_zero;
end
