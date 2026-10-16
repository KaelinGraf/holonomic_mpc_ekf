function r = test_ekf_noise_free()
%TEST_EKF_NOISE_FREE With no noise and no bias the filter must track the truth to integration accuracy.
%   Catches: frame and sign errors in the propagation (a wrong rotation gives errors
%   of tens of centimetres, not millimetres), wrong delay bookkeeping, broken wrap.
%   The residual is the Euler integration error of the staged approach: about 1 mm in
%   position and 5 mm/s in velocity while the robot accelerates at 0.06 m/s^2.
r.name = 'EKF noise-free tracking (frames, signs, delay bookkeeping)';
prm = dock_params('fullscale');
prm.imu.sigma_a = 0;  prm.imu.sigma_r = 0;  prm.imu.bias_a = [0; 0];  prm.imu.bias_r = 0;
prm.fid.sigma_p = 1e-4;  prm.fid.sigma_th = 1e-4;     % tiny but non-zero so S stays well conditioned
prm.ekf.Sigma_u = zeros(3);
prm.ekf.P0_handover = 1e-6*eye(5);                    % start essentially on the truth

tr   = make_truth_trajectory(prm.imu.T);
sens = simulate_sensors(tr, prm, 1);
out  = run_ekf(sens, prm, tr, struct('delay_comp', true, 'init', 'handover'));

ep  = max(hypot(out.err(:,1), out.err(:,2)));
eth = max(abs(out.err(:,3)));
ev  = max(hypot(out.err(:,4), out.err(:,5)));
r.pass = ep < 3e-3 && eth < deg2rad(0.05) && ev < 10e-3;
r.warn = false;
r.msg  = sprintf('max errors: position %.2f mm, heading %.3f deg, velocity %.1f mm/s (limits 3 mm, 0.05 deg, 10 mm/s)', ...
                 ep*1e3, rad2deg(eth), ev*1e3);
end
