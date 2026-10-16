function [M, gam] = mpc_constraints(F, Phi, xp, u_prev, lim, Jinv, Nc, Np)
%MPC_CONSTRAINTS Rows of M*dU <= gam in the deployed order (mpc.py constraints), 20 Nc + 3 Np.
%   increments   |du| <= du_max                                       6 Nc (hard)
%   amplitudes   per axis |u_b| <= v_max, |v_b| <= v_lat, or the L1
%                facets |u_b| + |v_b| <= v_max (lim.l1), |r_b| <= r_max 6 Nc (hard)
%   wheels       |Jinv u| <= w_max                                     8 Nc (hard)
%   penetration  x(k+j) <= x_max                                       Np   (soft)
%   heading      |theta(k+j)| <= th_max (180 deg: off)                 2 Np (soft)
m = 3;
C1 = kron(ones(Nc, 1), eye(m));
C2 = kron(tril(ones(Nc)), eye(m));
if lim.l1
    Aamp = [1 1 0; 1 -1 0; -1 1 0; -1 -1 0; 0 0 1; 0 0 -1];
    caps = [lim.v_max; lim.v_max; lim.v_max; lim.v_max; lim.r_max; lim.r_max];
else
    Aamp = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
    caps = [lim.v_max; lim.v_max; lim.v_lat; lim.v_lat; lim.r_max; lim.r_max];
end
Ab = kron(eye(Nc), Aamp);  Jb = kron(eye(Nc), Jinv);
Sx = kron(eye(Np), [1 0 0]);  Sth = kron(eye(Np), [0 0 1]);
Fx = F*xp(:);
M = [-eye(m*Nc); eye(m*Nc); Ab*C2; -Jb*C2; Jb*C2; Sx*Phi; Sth*Phi; -Sth*Phi];
gam = [repmat(lim.du_max(:), 2*Nc, 1);
       repmat(caps - Aamp*u_prev(:), Nc, 1);
       lim.w_max + Jb*C1*u_prev(:);  lim.w_max - Jb*C1*u_prev(:);
       lim.x_max - Sx*Fx;  lim.th_max - Sth*Fx;  lim.th_max + Sth*Fx];
end
