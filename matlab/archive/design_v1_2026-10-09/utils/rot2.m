function R = rot2(th)
%ROT2 Planar rotation matrix R(th) = [cos -sin; sin cos].
c = cos(th); s = sin(th);
R = [c, -s; s, c];
end
