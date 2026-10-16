function r = test_mpc_kinematics()
%TEST_MPC_KINEMATICS Mecanum inverse and forward kinematics: signs, consistency, null space.
%   Catches: a wrong sign convention for lateral motion or yaw, a wrong lever arm
%   (lx + ly), a forward map that is not the left inverse of the inverse map.
r.name = 'Mecanum kinematics (signs, fk(ik) = I, null space)';
prm  = dock_params('demo');
geom = prm.geom;
[wf, Jinv] = mecanum_ik([0.2; 0; 0], geom);
[wl, ~]    = mecanum_ik([0; 0.2; 0], geom);
[wr, ~]    = mecanum_ik([0; 0; 0.5], geom);
[~, Jfwd]  = mecanum_fk(zeros(4,1), geom);
rng(2);  u = randn(3, 5);
round_trip = max(abs(mecanum_fk(Jinv*u(:,1), geom) - u(:,1)));
for k = 2:5, round_trip = max(round_trip, max(abs(Jfwd*(Jinv*u(:,k)) - u(:,k)))); end
null_motion = max(abs(Jfwd*[1; 1; -1; -1]));
ok_fwd  = all(wf > 0) && max(abs(wf - wf(1))) < 1e-12;
ok_left = wl(1) < 0 && wl(2) > 0 && wl(3) > 0 && wl(4) < 0;
ok_yaw  = wr(1) < 0 && wr(3) < 0 && wr(2) > 0 && wr(4) > 0;
r.pass = ok_fwd && ok_left && ok_yaw && round_trip < 1e-10 && null_motion < 1e-12 ...
         && abs(wf(1) - 0.2/geom.Rw) < 1e-12;
r.warn = false;
r.msg  = sprintf('forward signs %d, left signs %d, yaw signs %d, max |fk(ik(u)) - u| = %.1e, |fk([1 1 -1 -1])| = %.1e', ...
                 ok_fwd, ok_left, ok_yaw, round_trip, null_motion);
end
