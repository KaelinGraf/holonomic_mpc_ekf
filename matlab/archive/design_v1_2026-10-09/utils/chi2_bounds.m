function [lo, hi] = chi2_bounds(k, alpha)
%CHI2_BOUNDS Two-sided (1-alpha) interval of a chi-square variable with k dof.
%   Wilson-Hilferty approximation (no Statistics Toolbox needed). For k = 5,
%   alpha = 0.05 it gives [0.80, 12.82] against the exact [0.83, 12.83].
if nargin < 2, alpha = 0.05; end
z  = sqrt(2) * erfinv(1 - alpha);          % two-sided normal quantile
a  = 2/(9*k);
lo = k * (1 - a - z*sqrt(a))^3;
hi = k * (1 - a + z*sqrt(a))^3;
end
