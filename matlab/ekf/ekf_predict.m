function [x, P, F_k, G_k] = ekf_predict(x, P, u, T, Sigma_u, Qfloor)
%EKF_PREDICT One IMU-driven prediction step of the docking EKF.
%   [x, P, F, G] = EKF_PREDICT(x, P, u, T, Sigma_u, Qfloor)
%
%   Inputs
%     x        [5x1]  state [px; py; th; vx; vy]: position and velocity in the
%                     target frame T, heading th in radians
%     P        [5x5]  state covariance
%     u        [3x1]  IMU input [ax; ay; r]: gravity-compensated body-frame
%                     acceleration [m/s^2] and yaw rate [rad/s]
%     T        [1x1]  IMU period [s]
%     Sigma_u  [3x3]  per-sample covariance of the IMU inputs
%     Qfloor   [5x5]  additive process-noise floor
%
%   Outputs
%     x, P     predicted state and covariance
%     F        [5x5]  df/dx, evaluated at the INPUT state (the linearisation point)
%     G        [5x3]  df/du, evaluated at the INPUT state
%
%   Equations: design/ekf_equations.pdf steps 3-6 (Euler map, Jacobians, Q_k, predictor).

th = x(3);
F_k = [1, 0, 0,                                     T, 0;
       0, 1, 0,                                     0, T;
       0, 0, 1,                                     0, 0;
       0, 0, -T*(sin(th)*u(1) + cos(th)*u(2)),      1, 0;
       0, 0,  T*(cos(th)*u(1) - sin(th)*u(2)),      0, 1];

G_k = [0,         0,          0;
       0,         0,          0;
       0,         0,          T;
       T*cos(th), -T*sin(th), 0;
       T*sin(th),  T*cos(th), 0];

Q_k = G_k * Sigma_u * G_k' + Qfloor;
P = F_k * P * F_k' + Q_k;
P = (P + P')/2;

x = [x(1) + T*x(4);
     x(2) + T*x(5);
     x(3) + T*u(3);
     x(4) + T*(cos(th)*u(1) - sin(th)*u(2));
     x(5) + T*(sin(th)*u(1) + cos(th)*u(2))];
x(3) = wrap_pi(x(3));
end
