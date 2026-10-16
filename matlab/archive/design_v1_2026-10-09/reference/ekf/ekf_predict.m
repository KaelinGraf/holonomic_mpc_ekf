function [x, P, F, G] = ekf_predict(x, P, u, T, Sigma_u, Qfloor)
%EKF_PREDICT [REFERENCE] One IMU-driven prediction step of the docking EKF.
%   Sealed reference implementation. Compare against it only after writing your own.
%   Interface: see matlab/ekf/ekf_predict.m. Equations: docking_formulation.pdf 3.2.

th = x(3); c = cos(th); s = sin(th);
ax = u(1); ay = u(2); r = u(3);

F = eye(5);
F(1,4) = T;
F(2,5) = T;
F(4,3) = -T*(s*ax + c*ay);
F(5,3) =  T*(c*ax - s*ay);

G = zeros(5,3);
G(3,3) = T;
G(4,1) = T*c;  G(4,2) = -T*s;
G(5,1) = T*s;  G(5,2) =  T*c;

x = x + T*[x(4); x(5); r; c*ax - s*ay; s*ax + c*ay];
x(3) = wrap_pi(x(3));

Q = G*Sigma_u*G' + Qfloor;
P = F*P*F' + Q;
P = (P + P')/2;
end
