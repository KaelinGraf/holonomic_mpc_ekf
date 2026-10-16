function r = test_mpc_prediction()
%TEST_MPC_PREDICTION F and Phi against a brute-force simulation of the augmented model,
%   and against the course's mpcgain.m for the decoupled SISO case at zero heading.
%   Catches: block indexing errors in Phi, powers of Ap off by one, wrong augmented
%   matrices, control horizon not zero-padded beyond Nc.
r.name = 'MPC prediction matrices (brute force, and the course''s mpcgain.m)';
prm = dock_params('fullscale');
Ts = prm.mpc.Ts;  Np = 12;  Nc = 4;
rng(3);
xc = [-0.8; 0.1; 0.3];  u0 = [0.2; 0.05; -0.1];
[A, B] = mpc_linearise(xc, u0, Ts);
[F, Phi, Ap, Bp, Cp] = mpc_predict_matrices(A, B, Np, Nc);
xp0 = randn(6, 1);  dU = randn(3*Nc, 1);
Y_sim = zeros(3*Np, 1);  xp = xp0;
for i = 1:Np
    if i <= Nc, du = dU((i-1)*3 + (1:3)); else, du = zeros(3,1); end
    xp = Ap*xp + Bp*du;
    Y_sim((i-1)*3 + (1:3)) = Cp*xp;
end
err_bf = max(abs(F*xp0 + Phi*dU - Y_sim));
ref_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'ref', 'MATLAB_e_amples_export', 'Sec5 MPC');
addpath(ref_dir);
[A0, B0] = mpc_linearise([0; 0; 0], zeros(3,1), Ts);
[F0, Phi0] = mpc_predict_matrices(A0, B0, Np, Nc);
[PhiPhi_c, PhiF_c] = mpcgain(1, Ts, 1, Nc, Np);
Fx   = F0(1:3:end, [1, 4]);
Phix = Phi0(1:3:end, 1:3:end);
err_course = max([max(abs(Phix'*Phix - PhiPhi_c), [], 'all'), max(abs(Phix'*Fx - PhiF_c), [], 'all')]);
sizes_ok = isequal(size(F), [3*Np, 6]) && isequal(size(Phi), [3*Np, 3*Nc]) && isequal(size(Ap), [6, 6]);
r.pass = err_bf < 1e-9 && err_course < 1e-9 && sizes_ok;
r.warn = false;
r.msg  = sprintf('brute-force mismatch %.1e, mismatch vs mpcgain.m (SISO, zero heading) %.1e, sizes ok %d', ...
                 err_bf, err_course, sizes_ok);
end
