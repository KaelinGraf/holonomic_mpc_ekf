# EKF output (brief Section 5). Published by ekf_node after every IMU sample (200 Hz).

std_msgs/Header header

float64[5] x
float64[5] p_diag
float64 r_gyro
float64 imu_rate
float64 fix_age
float64 last_nis
bool last_fix_accepted
