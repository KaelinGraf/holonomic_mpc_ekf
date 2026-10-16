function [x, P] = ekf_zv(x, P, Tk, f)
%EKF_ZV Zero-velocity reset at the end of a filter step (ekf.py zero_velocity()).
%   Applied while every command component has been within 1e-4 of zero for 0.3 s:
%   velocity set to zero, its rows and columns of P cleared, velocity variance
%   zv_sigma_v^2, and the position and heading variances grown by zv_pos_rate and
%   zv_th_rate per second of step length.
x(4:5) = 0;
P(4:5, :) = 0;
P(:, 4:5) = 0;
P(4, 4) = f.zv_sigma_v^2;
P(5, 5) = f.zv_sigma_v^2;
P(1, 1) = P(1, 1) + f.zv_pos_rate*Tk;
P(2, 2) = P(2, 2) + f.zv_pos_rate*Tk;
P(3, 3) = P(3, 3) + f.zv_th_rate*Tk;
end
