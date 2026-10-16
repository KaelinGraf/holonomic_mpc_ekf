function [w, Jinv] = mecanum_ik(u, geom)
%MECANUM_IK [REFERENCE] Body velocity to wheel speeds, FL FR RL RR, rollers at 45 deg.
L = geom.lx + geom.ly;
Jinv = (1/geom.Rw) * [1, -1, -L;
                      1,  1,  L;
                      1,  1, -L;
                      1, -1,  L];
w = Jinv * u(:);
end
