function [F, Phi, Ap, Bp, Cp] = mpc_predict_matrices(A, B, Np, Nc)
%MPC_PREDICT_MATRICES Augmented incremental model and prediction matrices.
%   [F, Phi, Ap, Bp, Cp] = MPC_PREDICT_MATRICES(A, B, Np, Nc)
%   A, B     [3x3]  per-tick linearisation from mpc_linearise; the output matrix is C = eye(3)
%   Np, Nc   prediction and control horizons
%   Ap       [6x6]  [A 0; A I]        augmented state x_p = [dx; y], course Section 5.3.2 with C = I
%   Bp       [6x3]  [B; B]
%   Cp       [3x6]  [0 I]
%   F        [3Np x 6]   rows Cp*Ap^i, i = 1..Np
%   Phi      [3Np x 3Nc] block (i, j) = Cp*Ap^(i-j)*Bp for i >= j, zero above the diagonal
%
%   so that Y = F*x_p + Phi*dU stacks the predicted outputs y(k+1|k) .. y(k+Np|k).
%   Reference: design doc eq. (24)-(26); the course's mpcgain.m is the SISO version.
%   Checked by tests/test_mpc_prediction (brute-force simulation of the augmented model)
%   and against mpcgain.m for the decoupled SISO case at zero heading.

n = size(A, 1);  m = size(B, 2);  p = n;
Ap = [A, zeros(n, p); A, eye(p)];        % C = I so C*A = A, C*B = B
Bp = [B; B];
Cp = [zeros(p, n), eye(p)];

F   = zeros(p*Np, n+p);
Phi = zeros(p*Np, m*Nc);
h   = cell(Np, 1);                       % h{i} = Cp*Ap^(i-1)*Bp
CpA = Cp;                                % Cp*Ap^(i-1)
for i = 1:Np
    h{i} = CpA*Bp;
    CpA  = CpA*Ap;                       % now Cp*Ap^i
    F((i-1)*p + (1:p), :) = CpA;
end
for i = 1:Np
    for j = 1:min(i, Nc)
        Phi((i-1)*p + (1:p), (j-1)*m + (1:m)) = h{i-j+1};
    end
end
end
