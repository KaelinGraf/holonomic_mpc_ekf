function r = test_ekf_jacobians()
%TEST_EKF_JACOBIANS Finite-difference check of F = df/dx and G = df/du from ekf_predict.
%   Catches: Jacobian evaluated at the wrong point, missing or wrong-sign terms,
%   degrees/radians mix-ups. The propagation itself is checked by test_ekf_noise_free.
r.name = 'EKF Jacobians against finite differences';
prm = dock_params('fullscale');
T = prm.imu.T;
x0 = [-1.2; 0.3; 0.4; 0.15; -0.05];
u0 = [0.3; -0.2; 0.1];
P0 = 1e-3*eye(5);

[~, ~, F, G] = ekf_predict(x0, P0, u0, T, prm.ekf.Sigma_u, prm.ekf.Qfloor);
f = @(x, u) ekf_predict(x, P0, u, T, prm.ekf.Sigma_u, prm.ekf.Qfloor);

h = 1e-6;
Fn = zeros(5, 5);
for i = 1:5
    e = zeros(5, 1); e(i) = h;
    d = f(x0 + e, u0) - f(x0 - e, u0);  d(3) = wrap_pi(d(3));
    Fn(:, i) = d / (2*h);
end
Gn = zeros(5, 3);
for i = 1:3
    e = zeros(3, 1); e(i) = h;
    d = f(x0, u0 + e) - f(x0, u0 - e);  d(3) = wrap_pi(d(3));
    Gn(:, i) = d / (2*h);
end
errF = max(abs(F(:) - Fn(:)));
errG = max(abs(G(:) - Gn(:)));
r.pass = errF < 1e-6 && errG < 1e-6;
r.warn = false;
r.msg  = sprintf('max |F - F_fd| = %.1e, max |G - G_fd| = %.1e (limit 1e-6)', errF, errG);
end
