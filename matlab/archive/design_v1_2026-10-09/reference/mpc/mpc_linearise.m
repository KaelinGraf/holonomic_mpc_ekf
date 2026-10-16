function [A, B] = mpc_linearise(xc, u_prev, Ts)
%MPC_LINEARISE [REFERENCE] Jacobians of f_c(x,u) = x + Ts*[R(th)*u(1:2); u(3)] at (xc, u_prev).
th = xc(3);  c = cos(th);  s = sin(th);
ub = u_prev(1);  vb = u_prev(2);
A = eye(3);
A(1,3) = -Ts*(ub*s + vb*c);
A(2,3) =  Ts*(ub*c - vb*s);
B = Ts*[c, -s, 0;
        s,  c, 0;
        0,  0, 1];
end
