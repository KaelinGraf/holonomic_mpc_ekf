function [M, gam] = mpc_constraints(F, Phi, xp, u_prev, lim, Jinv, Nc, Np)
%MPC_CONSTRAINTS Assemble the linear inequality constraints M*dU <= gam.
%   [M, gam] = MPC_CONSTRAINTS(F, Phi, xp, u_prev, lim, Jinv, Nc, Np)
%   F, Phi   prediction matrices for the current tick
%   xp       [6x1]  current augmented state [dx; y]
%   u_prev   [3x1]  last applied command
%   lim      struct: du_max [3x1]; scalars v_max, r_max, w_max, x_max, th_max
%   Jinv     [4x3]  mecanum inverse kinematics (wheel speeds = Jinv*u)
%   Nc, Np   horizons
%
%   Rows, in this order (design doc eq. (33)-(37), course Section 5.4.2):
%     increments   -dU <= dUmax,  dU <= dUmax                      2*3Nc rows   (hard)
%     amplitudes   on U = C1*u_prev + C2*dU: the translational cap |u_b| + |v_b| <= v_max
%                  as its four facets [1 1 0; 1 -1 0; -1 1 0; -1 -1 0]*u <= v_max, and
%                  the yaw-rate cap |r_b| <= r_max as two rows            6Nc rows   (hard)
%     wheel speeds -w_max <= (I_Nc kron Jinv)*U <= w_max            2*4Nc rows   (hard)
%     penetration  Sx*(F*xp + Phi*dU) <= x_max,  Sx = I_Np kron [1 0 0]     Np rows (soft)
%     heading      |Sth*(F*xp + Phi*dU)| <= th_max, Sth = I_Np kron [0 0 1]  2Np rows (soft)
%   with C1 = ones(Nc,1) kron eye(3) and C2 = tril(ones(Nc)) kron eye(3). The speed cap is
%   an L1 bound rather than a box on purpose: a box is 41 % faster along its diagonal and
%   the optimiser yaws the body to use it (23 deg on the demonstrator); the L1 set is the
%   shape of the wheel-speed limit at zero yaw rate, so the chassis is fastest along its
%   own axes and has no reason to yaw. The heading corridor keeps the marker in the
%   camera's view. mpc_step treats the last 3Np rows as soft. Total 20*Nc + 3*Np rows by
%   3Nc columns. Checked by tests/test_mpc_constraints.

m = 3;
I_c = eye(m*Nc);
C1  = kron(ones(Nc, 1), eye(m));
C2  = kron(tril(ones(Nc)), eye(m));
dUmax = kron(ones(Nc, 1), lim.du_max(:));
D     = kron(eye(Nc), [1 1 0; 1 -1 0; -1 1 0; -1 -1 0]);   % L1 facets on (u_b, v_b)
Sr    = kron(eye(Nc), [0 0 1]);                            % yaw rate
Vmax  = lim.v_max * ones(4*Nc, 1);
Rmax  = lim.r_max * ones(Nc, 1);
Jb    = kron(eye(Nc), Jinv);
Wmax  = lim.w_max * ones(4*Nc, 1);
Sx    = kron(eye(Np), [1, 0, 0]);
Sth   = kron(eye(Np), [0, 0, 1]);
M   = [-I_c;  I_c;  D*C2;  -Sr*C2;  Sr*C2;  -Jb*C2;  Jb*C2;  Sx*Phi;  Sth*Phi;  -Sth*Phi];
gam = [dUmax; dUmax;
       Vmax - D*C1*u_prev(:);  Rmax + Sr*C1*u_prev(:);  Rmax - Sr*C1*u_prev(:);
       Wmax + Jb*C1*u_prev(:);  Wmax - Jb*C1*u_prev(:);
       lim.x_max*ones(Np, 1) - Sx*F*xp(:);
       lim.th_max*ones(Np, 1) - Sth*F*xp(:);
       lim.th_max*ones(Np, 1) + Sth*F*xp(:)];
end
