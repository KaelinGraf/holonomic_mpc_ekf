function [A, B] = mpc_linearise(xc, u_prev, Ts)
%MPC_LINEARISE Per-tick linearisation of the kinematic plant model.
%   [A, B] = MPC_LINEARISE(xc, u_prev, Ts)
%   xc      [3x1]  current pose estimate [x; y; th] in the target frame
%   u_prev  [3x1]  last applied body-velocity command [u_b; v_b; r_b]
%   Ts      MPC period [s]
%   A       [3x3]  Jacobian of the discrete kinematic map f_c with respect to the pose
%   B       [3x3]  Jacobian with respect to the command
%
%   Reference: design/docking_formulation.pdf, eq. (21) for f_c and (23) for A_k, B_k.
%   f_c(x, u) = x + Ts * [u_b cos th - v_b sin th;  u_b sin th + v_b cos th;  r_b],
%   linearised about (xc, u_prev). B is Ts times the rotation into the target frame.
%   Checked by tests/test_mpc_linearisation against finite differences.

th = xc(3);  c = cos(th);  s = sin(th);
ub = u_prev(1);  vb = u_prev(2);
A = eye(3);
A(1,3) = -Ts*(ub*s + vb*c);
A(2,3) =  Ts*(ub*c - vb*s);
B = Ts*[c, -s, 0;
        s,  c, 0;
        0,  0, 1];
end
