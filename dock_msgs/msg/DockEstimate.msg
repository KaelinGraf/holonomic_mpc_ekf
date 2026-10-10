# EKF output (brief Section 5). Published by ekf_node after every IMU sample (200 Hz).

std_msgs/Header header    # stamp = IMU sample time

float64[5] x              # [x, y, theta, vx, vy] in target [m, m, rad, m/s, m/s]
float64[5] p_diag         # diagonal of P
float64 r_gyro            # yaw rate of the sample [rad/s]
float64 fix_age           # seconds since the last accepted fix
float64 last_nis          # nu' S^-1 nu of the last fix
bool last_fix_accepted
