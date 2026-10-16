function r = test_ekf_delay()
%TEST_EKF_DELAY Delay compensation removes the v*tau_d lag; without it the lag must appear.
%   Constant-velocity cruise along +x at heading zero. Without compensation the
%   estimate trails the truth by about v*tau_d (10 mm at 0.2 m/s); with the ring
%   buffer the mean error must be a small fraction of that.
%   Catches: buffer indexed by availability time instead of capture time, or the
%   innovation formed against the current state while claiming compensation.
r.name = 'Fiducial delay compensation (constant-velocity cruise)';
prm = dock_params('fullscale');

v = 0.20;  T = prm.imu.T;  t = (0:T:20)';  N = numel(t);
tr.T = T;  tr.t = t;
tr.pose = [-3 + v*t, zeros(N, 2)];
tr.vel  = [v*ones(N, 1), zeros(N, 2)];
tr.acc  = zeros(N, 3);  tr.ab = zeros(N, 2);  tr.r = zeros(N, 1);

sens = simulate_sensors(tr, prm, 3);
on  = run_ekf(sens, prm, tr, struct('delay_comp', true));
off = run_ekf(sens, prm, tr, struct('delay_comp', false));

m = t > 5;                                   % after convergence
lag_expected = v * prm.fid.delay;            % the estimate trails the truth: err = x_hat - x < 0
lag_off = mean(off.err(m, 1));
lag_on  = mean(on.err(m, 1));
r.pass = abs(-lag_off - lag_expected) < 0.3*lag_expected && abs(lag_on) < 0.2*lag_expected;
r.warn = false;
r.msg  = sprintf('mean x error: off %.1f mm (expect about %.1f mm), on %.1f mm (limit %.1f mm)', ...
                 lag_off*1e3, -lag_expected*1e3, lag_on*1e3, 0.2*lag_expected*1e3);
end
