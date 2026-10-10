# Fiducial pseudo-measurement z = [x, y, theta] of base_link in the target frame
# (brief Section 2): the inference pipeline's board pose after the chain.
# Published by ekf_node for every pose it fuses or rejects (15 Hz), for logging
# and for the supervisor's roll and pitch monitor.

std_msgs/Header header    # stamp = capture time (image stamp minus the exposure offset)

float64 x                 # z in the target frame [m]
float64 y                 # [m]
float64 theta             # [rad]
float64[9] covariance     # R of [x, y, theta], row-major: the pose's live covariance through the chain
bool covariance_valid     # false: range outside the pipeline's fitted noise model (its clamped value used)
float64 roll              # monitored only [rad]
float64 pitch             # monitored only [rad]
float64 range             # |t_CM| [m]
bool ambiguous            # two IPPE solutions; this is the one nearer the prediction
