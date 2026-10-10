# MPC diagnostics (brief Section 6). Published by mpc_node every tick (20 Hz).

std_msgs/Header header    # stamp = stamp of the estimate the tick used

float64[3] u              # command sent on cmd_vel [u_b, v_b, r_b]
float64[] y_pred          # Y = F x_p + Phi dU as Np rows of [x, y, theta], flattened (3*Np)
float64 slack
int32 exit_flag           # the solver's own code: DAQP exitflag or OSQP status_val
float64 solve_ms
bool fallback
