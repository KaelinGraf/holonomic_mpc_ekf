function f = ekf_advance(f, t, u, zv)
%EKF_ADVANCE Advance the filter to one IMU sample's stamp (ekf.py Ekf.advance).
%   The interval since the last step is split into n = max(1, round(dt/T0)) equal
%   steps holding u; the floor is added as (h/T0) Q_floor per step. A repeated stamp
%   advances nothing; a gap longer than buffer_s (0.5 s) advances buffer_s and the clock
%   is reset to t. Nothing is buffered before the filter starts (first pose).
if ~f.started || t <= f.t_last, return; end
t0 = f.t_last;
gap = t - t0 > f.buffer_s;
if gap, tend = t0 + f.buffer_s; else, tend = t; end
n = max(1, round((tend - t0) / f.T0));
h = (tend - t0) / n;
for k = 1:n
    f = ekf_step(f, u, h, zv);
    f = ekf_push(f, t0 + k*h, u, h, zv);
end
if gap, f.t_last = t; else, f.t_last = tend; end
end
