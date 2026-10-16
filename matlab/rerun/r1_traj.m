function tr = r1_traj(prm)
%R1_TRAJ Analytic truth for the filter-alone runs: 2.5 s at rest (zeroing), then a quintic
%   from (-0.85, 0.20 m) looking at the board to the standoff in 8 s, 1 s hold, to the dock
%   in 8 s, 2 s hold. All fixes inside the half-field and 1.2 m.
g = prm.geom;
p0 = [-0.85; 0.20; 0];
p0(3) = atan2(g.board(2) - p0(2), g.board(1) - (p0(1) + g.cam(1)));
segs = [struct('pf', p0, 'tf', 2.5, 'hold', 0), ...
        struct('pf', [-prm.sup.standoff; 0; 0], 'tf', 8, 'hold', 1), ...
        struct('pf', [0; 0; 0], 'tf', 8, 'hold', 2)];
tr = make_truth_trajectory(prm.imu.T, p0, segs);
end
