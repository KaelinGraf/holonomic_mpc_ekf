function [F, Phi, Ap, Bp, Cp] = mpc_predict_matrices(A, B, Np, Nc)
%MPC_PREDICT_MATRICES [REFERENCE] Augmented incremental model (C = I) and prediction matrices.
n = size(A, 1);  m = size(B, 2);  p = n;
Ap = [A, zeros(n, p); A, eye(p)];
Bp = [B; B];
Cp = [zeros(p, n), eye(p)];

F   = zeros(p*Np, n+p);
Phi = zeros(p*Np, m*Nc);
h   = cell(Np, 1);
CpA = Cp;
for i = 1:Np
    h{i} = CpA*Bp;
    CpA  = CpA*Ap;
    F((i-1)*p + (1:p), :) = CpA;
end
for i = 1:Np
    for j = 1:min(i, Nc)
        Phi((i-1)*p + (1:p), (j-1)*m + (1:m)) = h{i-j+1};
    end
end
end
