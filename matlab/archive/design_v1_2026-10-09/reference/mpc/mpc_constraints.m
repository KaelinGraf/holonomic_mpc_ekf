function [M, gam] = mpc_constraints(F, Phi, xp, u_prev, lim, Jinv, Nc, Np)
%MPC_CONSTRAINTS [REFERENCE] Increment, amplitude, wheel-speed rows (hard) followed by the
%   no-penetration and heading-corridor rows (soft in mpc_step). 20*Nc + 3*Np rows.
%   The translational speed cap is the L1 bound |u_b| + |v_b| <= v_max (four rows per
%   step), not a box: a box is 41 % faster along its diagonal, which the optimiser
%   exploits by yawing the body. The L1 set is the shape of the wheel-speed limit at
%   zero yaw rate, so the chassis is fastest along its own axes and has no reason to yaw.
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
