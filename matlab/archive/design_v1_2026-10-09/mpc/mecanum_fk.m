function [u, Jfwd] = mecanum_fk(w, geom)
%MECANUM_FK Forward kinematics: wheel speeds to body velocity (left pseudo-inverse).
%   [u, Jfwd] = MECANUM_FK(w, geom)
%   w     [4x1]  wheel angular speeds FL, FR, RL, RR [rad/s]
%   geom  struct with Rw, lx, ly
%   u     [3x1]  body velocity [u_b; v_b; r_b]
%   Jfwd  [3x4]  the matrix with u = Jfwd * w, satisfying Jfwd * Jinv = eye(3)
%
%   Reference: Appendix A. The null space of Jfwd is spanned by [1; 1; -1; -1], the
%   wheel combination that produces no chassis motion.

L = geom.lx + geom.ly;
Jfwd = (geom.Rw/4) * [ 1,    1,    1,    1;
                      -1,    1,    1,   -1;
                      -1/L,  1/L, -1/L,  1/L];
u = Jfwd * w(:);
end
