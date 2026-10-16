function [w, Jinv] = mecanum_ik(u, geom)
%MECANUM_IK Inverse kinematics: body velocity to wheel speeds.
%   [w, Jinv] = MECANUM_IK(u, geom)
%   u     [3x1]  body velocity [u_b; v_b; r_b]: forward, left, yaw rate (counter-clockwise)
%   geom  struct with Rw (wheel radius), lx (half wheelbase), ly (half track)
%   w     [4x1]  wheel angular speeds in the order FL, FR, RL, RR [rad/s]
%   Jinv  [4x3]  the matrix with w = Jinv * u
%
%   Reference: design/docking_formulation.pdf, Appendix A, rollers at 45 degrees.
%   Contract checked by tests/test_mpc_kinematics: pure forward motion drives all four
%   wheels positive; pure left motion drives FL and RR negative, FR and RL positive;
%   pure counter-clockwise yaw drives the left wheels negative and the right wheels
%   positive; mecanum_fk(mecanum_ik(u)) returns u.

L = geom.lx + geom.ly;
Jinv = (1/geom.Rw) * [1, -1, -L;
                      1,  1,  L;
                      1,  1, -L;
                      1, -1,  L];
w = Jinv * u(:);
end
