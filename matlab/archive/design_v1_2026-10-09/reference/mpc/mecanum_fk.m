function [u, Jfwd] = mecanum_fk(w, geom)
%MECANUM_FK [REFERENCE] Wheel speeds to body velocity, left pseudo-inverse of mecanum_ik.
L = geom.lx + geom.ly;
Jfwd = (geom.Rw/4) * [ 1,    1,    1,    1;
                      -1,    1,    1,   -1;
                      -1/L,  1/L, -1/L,  1/L];
u = Jfwd * w(:);
end
