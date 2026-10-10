# Fiducial pseudo-measurement z = [x, y, theta] of base_link in the target frame
# (brief Section 2): the inference pipeline's board pose after the chain.
# Published by ekf_node for every pose it receives (15 Hz), for logging and for
# the supervisor's roll and pitch monitor.

std_msgs/Header header

float64 x
float64 y
float64 theta
float64 sigma_p
float64 sigma_th
float64 roll
float64 pitch
float64 range
int32 marker_id
bool valid
