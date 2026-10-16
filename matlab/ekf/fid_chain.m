function [z, J] = fid_chain(meas, geom)
%FID_CHAIN Fiducial pre-transform: camera-frame board pose to the pose pseudo-measurement.
%   [z, J] = FID_CHAIN(meas, geom)
%   meas  [3x1] board pose in the camera's ground-plane frame: [t_f; t_l; psi], the board
%         centre forward and left of the camera [m] and the yaw of the board's outward
%         normal relative to the camera's optical axis, both laid on the ground plane
%   geom  .cam [2x1] camera in the body frame, .board [2x1] board centre in the target
%         frame, .psi_n yaw of the board's outward normal in the target frame
%   z     [3x1] chassis pose in the target frame [x; y; theta]
%   J     [3x3] dz/dmeas, so that R_z = J * Sigma * J' for a camera-frame covariance Sigma
%
%   theta_z = psi_n - psi,  p_z = m - R(theta_z) (t + c)       (design doc Eq. 5, planar)
%   The board's in-plane rotation and the vertical components do not enter.
t  = meas(1:2);
th = wrap_pi(geom.psi_n - meas(3));
c = cos(th);  s = sin(th);
R = [c, -s; s, c];
v = t(:) + geom.cam(:);
z = [geom.board(:) - R*v; th];
if nargout > 1
    J = [-R, R*[-v(2); v(1)]; 0, 0, -1];      % d p_z/d psi = R*S*v, S = [0 -1; 1 0]
end
end
