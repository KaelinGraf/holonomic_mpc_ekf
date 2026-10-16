function f = ekf_step(f, u, h, zv)
%EKF_STEP One Euler step of h holding u, with the floor scaled by h/T0 and the
%   zero-velocity reset at its end when zv (ekf.py _step + zero_velocity).
[f.x, f.P] = ekf_predict(f.x, f.P, u, h, f.Sigma_u, (h / f.T0) * f.Qfloor);
if zv, [f.x, f.P] = ekf_zv(f.x, f.P, h, f); end
end
