function [st, imu] = plant_step(st, w_cmd, prm, dt)
%PLANT_STEP Truth plant: mecanum chassis with wheel-speed lag and slip, one step of dt.
%   [st, imu] = PLANT_STEP(st, w_cmd, prm, dt)
%   st     state struct: pose [3x1] in the target frame, vb [3x1] actual body velocity,
%          w [4x1] actual wheel speeds, slip [2x1] body-velocity offset the wheels do not see
%   w_cmd  [4x1] commanded wheel speeds (held between MPC ticks)
%   imu    struct: a [2x1] body-frame acceleration of the chassis centre (gravity-free truth),
%          r yaw rate; add noise and bias outside
%
%   Wheels follow their commands through a first-order lag with time constant
%   prm.plant.tau; the chassis moves at the wheel-implied body velocity plus a slowly
%   varying slip offset (random walk, std prm.plant.slip_sigma, bounded). The IMU truth
%   is a_b = dv_b/dt + r*S*v_b, the body-frame acceleration of a point moving with
%   velocity v_b in a frame rotating at r. Initialise with plant_init.

L = prm.geom.lx + prm.geom.ly;
Jfwd = (prm.geom.Rw/4) * [1, 1, 1, 1; -1, 1, 1, -1; -1/L, 1/L, -1/L, 1/L];

vb_old = st.vb;
% wheel-speed loops
st.w = st.w + (dt/prm.plant.tau) * (w_cmd(:) - st.w);
% slip: bounded random walk on the body-velocity offset
st.slip = st.slip + sqrt(dt) * 0.1*prm.plant.slip_sigma * randn(2, 1);
st.slip = max(min(st.slip, 3*prm.plant.slip_sigma), -3*prm.plant.slip_sigma);
% actual body velocity
st.vb = Jfwd*st.w + [st.slip; 0];
% pose integration (midpoint velocity)
vmid = 0.5*(vb_old + st.vb);
th = st.pose(3);
R  = [cos(th), -sin(th); sin(th), cos(th)];
st.pose(1:2) = st.pose(1:2) + dt * R * vmid(1:2);
st.pose(3)   = wrap_pi(st.pose(3) + dt * vmid(3));
% IMU truth
r = st.vb(3);
imu.a = (st.vb(1:2) - vb_old(1:2))/dt + r*[-st.vb(2); st.vb(1)];
imu.r = r;
end
