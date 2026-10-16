function P = make_starts(prm, n, seed)
%MAKE_STARTS Random start poses inside the camera's working range (brief 2.1).
%   Uniform draws x in [-1.3, -standoff], y in [-0.4, 0.4] m, theta in [-20, 20] deg,
%   kept if the camera-to-board range is <= 1.2 m, the board centre is inside the
%   horizontal half-field and the start is at least the standoff from the docked pose.
rs = RandStream('mt19937ar', 'Seed', seed);
P = zeros(3, 0);
while size(P, 2) < n
    p = [-1.3 + (1.3 - prm.sup.standoff)*rand(rs); -0.4 + 0.8*rand(rs); deg2rad(-20 + 40*rand(rs))];
    fx = fiducial_measure(p, prm, zeros(3, 1), 0);
    if fx.valid && fx.range <= 1.2 && hypot(p(1), p(2)) >= prm.sup.standoff
        P(:, end+1) = p;
    end
end
end
