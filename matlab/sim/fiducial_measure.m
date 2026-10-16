function fx = fiducial_measure(pose, prm, e3, scale)
%FIDUCIAL_MEASURE One simulated fix of the board from the true chassis pose.
%   fx = FIDUCIAL_MEASURE(pose, prm, e3, scale)
%   e3     [3x1] standard normal draws for this fix;  scale  truth-noise multiplier
%   fx     .valid (board in the half-field and inside the range), .z, .R (the filter's
%          R_z), .range, .bearing
%   'law'  (full scale): the true board pose in the camera's ground-plane frame
%          [t_f; t_l; psi] gets an error drawn from the detector's range law
%          sigma_i = a_i z'^k_i, z' = z_scale * t_f / square (forward: optical z,
%          lateral: optical x, yaw: rotation about optical y); z = chain(meas) and
%          R_z = J Sigma J' through the chain's Jacobian, heading std floored at 1 deg.
%   'diag' (prototype): z = pose + noise with sigma_p(l) = sigma_p0 sqrt(1 + (l/d0)^2),
%          sigma_th; R_z = diag(sigma_p^2, sigma_p^2, max(sigma_th, floor)^2).
g = prm.geom;  F = prm.fid;
th = pose(3);  c = cos(th);  s = sin(th);
pc = pose(1:2) + [c*g.cam(1) - s*g.cam(2); s*g.cam(1) + c*g.cam(2)];
d = g.board(:) - pc;
t = [c*d(1) + s*d(2); -s*d(1) + c*d(2)];
fx.range = norm(t);
fx.bearing = atan2(t(2), t(1));
fx.valid = t(1) > 0 && abs(fx.bearing) <= F.hfov && fx.range <= F.range_max;
floor2 = prm.fid.sigma_th_floor^2;
switch F.model
    case 'law'
        psi = wrap_pi(g.psi_n - th);
        sig = law(t(1), F);
        meas = [t; psi] + scale*sig.*e3(:);
        [fx.z, J] = fid_chain(meas, g);
        R = J*diag(law(meas(1), F).^2)*J';
        R(3, 3) = max(R(3, 3), floor2);
        fx.R = 0.5*(R + R');
    case 'diag'
        sp = F.sigma_p0*sqrt(1 + (fx.range/F.d0)^2);
        fx.z = pose(:) + scale*[sp; sp; F.sigma_th].*e3(:);
        fx.z(3) = wrap_pi(fx.z(3));
        fx.R = diag([sp^2, sp^2, max(F.sigma_th^2, floor2)]);
end
end

function sig = law(tf, F)
zs = max(F.z_scale*tf/F.square, 1);
sig = [F.law_tr_a(3)*zs^F.law_tr_k(3)*F.square;
       F.law_tr_a(1)*zs^F.law_tr_k(1)*F.square;
       F.law_rot_a(2)*zs^F.law_rot_k(2)];
end
