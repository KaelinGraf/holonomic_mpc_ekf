function r = test_mpc_linearisation()
%TEST_MPC_LINEARISATION A_k, B_k against finite differences of the discrete kinematic map.
%   Catches: Jacobian of the continuous model used without the identity and Ts,
%   rotation transposed, linearisation point taken at u = 0 instead of u_prev.
r.name = 'MPC linearisation against finite differences';
prm = dock_params('fullscale');
Ts  = prm.mpc.Ts;
fc  = @(x, u) x + Ts*[u(1)*cos(x(3)) - u(2)*sin(x(3)); u(1)*sin(x(3)) + u(2)*cos(x(3)); u(3)];
xc  = [-1.1; 0.25; 0.35];  u0 = [0.22; -0.08; 0.12];
[A, B] = mpc_linearise(xc, u0, Ts);
h = 1e-6;  An = zeros(3);  Bn = zeros(3);
for i = 1:3
    e = zeros(3,1);  e(i) = h;
    An(:,i) = (fc(xc + e, u0) - fc(xc - e, u0)) / (2*h);
    Bn(:,i) = (fc(xc, u0 + e) - fc(xc, u0 - e)) / (2*h);
end
errA = max(abs(A(:) - An(:)));  errB = max(abs(B(:) - Bn(:)));
r.pass = errA < 1e-7 && errB < 1e-7;
r.warn = false;
r.msg  = sprintf('max |A - A_fd| = %.1e, max |B - B_fd| = %.1e (limit 1e-7)', errA, errB);
end
