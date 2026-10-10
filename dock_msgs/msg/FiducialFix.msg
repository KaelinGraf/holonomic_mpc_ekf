# Fiducial pseudo-measurement z = [x, y, theta] of base_link in the target frame
# (brief Section 2): the inference pipeline's board pose after the chain.
# Published by ekf_node for every pose it receives (15 Hz), for logging and for
# the supervisor's roll and pitch monitor.

std_msgs/Header header    # stamp = capture time (image stamp minus the exposure offset)

float64 x                 # z in the target frame [m]
float64 y                 # [m]
float64 theta             # [rad]
float64 sigma_p           # per-fix standard deviations: sigma_p(d) [m]
float64 sigma_th          # and sigma_th [rad]
float64 roll              # monitored only [rad]
float64 pitch             # monitored only [rad]
float64 range             # |t_CM| [m], for the noise model and logging
int32 marker_id
bool valid
