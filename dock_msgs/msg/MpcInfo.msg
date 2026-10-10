# MPC diagnostics (brief Section 6). Published by mpc_node every tick (20 Hz).

std_msgs/Header header

float64[3] u
float64[] y_pred
float64 slack
int32 exit_flag
float64 solve_ms
bool fallback
