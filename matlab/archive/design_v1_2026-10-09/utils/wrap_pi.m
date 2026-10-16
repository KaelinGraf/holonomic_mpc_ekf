function a = wrap_pi(a)
%WRAP_PI Wrap angle(s) in radians to the interval [-pi, pi).
a = mod(a + pi, 2*pi) - pi;
end
