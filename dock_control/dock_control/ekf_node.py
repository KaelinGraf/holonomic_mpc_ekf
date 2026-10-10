"""
EKF node (brief Section 5): IMU and board poses in, the docking estimate out.

Inputs come from the existing nodes (Conv-ChArT-Wireless-Inference):
imu/data from p4p_serial_bridge at the IMU rate, and inference_result from the
convchart_ros pipeline, the board pose in the camera frame stamped with the
Pi's image stamp. The fiducial chain runs inside the pose callback, so a pose
costs no extra node or topic hop. One single-threaded executor, so the IMU and
pose callbacks never overlap.

Publishes dock/estimate after every IMU sample, fiducial/fix for every pose (the
log and the supervisor's roll and pitch monitor), and target -> base_link on TF
at 20 Hz. Nothing is estimated until the start-up zeroing has finished and the
first pose has initialised the filter.

The filter steps in fixed T = 1/imu.rate over each sample's elapsed time
(Ekf.advance), so it stays right if the IMU drops samples or runs slow; the
measured IMU rate goes out in every estimate, and the supervisor will not start
moving unless it is within 10 % of imu.rate.
"""

import math

from builtin_interfaces.msg import Time
from convchart_interfaces.msg import RosInferenceResult
from convchart_qos.qos import INFERENCE_RESULT_QOS
from dock_control import params
from dock_control.ekf import Ekf
from dock_control.fiducial import FiducialChain
from dock_control.geometry import quat_to_rot
from dock_control.rosutil import run, seconds
from dock_msgs.msg import DockEstimate, FiducialFix
from geometry_msgs.msg import TransformStamped
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

TF_RATE = 20.0      # Hz, target -> base_link (brief Section 3)
ZERO_S = 2.0        # s of stationary IMU averaged into the accelerometer offset (brief Section 5)
RATE_TOL = 0.1      # warn when the measured IMU rate is off the configured one by more

# BEST_EFFORT matches a RELIABLE publisher (the bridge) and a sensor-data one (a
# dedicated IMU driver) alike, where RELIABLE would silently miss the second. On one
# machine nothing is lost; depth 10 keeps every sample through a slow callback.
IMU_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                     reliability=QoSReliabilityPolicy.BEST_EFFORT)
# The pipeline's own profile, but VOLATILE: replaying its latched (TRANSIENT_LOCAL)
# history at start-up would initialise the filter from stale poses.
POSE_QOS = QoSProfile(history=INFERENCE_RESULT_QOS.history, depth=INFERENCE_RESULT_QOS.depth,
                      reliability=INFERENCE_RESULT_QOS.reliability,
                      durability=QoSDurabilityPolicy.VOLATILE)


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        p = params.from_node(self)
        # Image stamp minus capture time: the Pi stamps a frame when it reads it,
        # after exposure and readout. Measure it once with the LED test (brief
        # Section 10.b, step 3).
        exposure_offset = self.declare_parameter('exposure_offset', 0.0).value

        # --- constant, built once ---
        self.ekf = Ekf(p)
        self.chain = FiducialChain(p)
        self.exposure_ns = round(exposure_offset * 1e9)
        self.rate = p['imu']['rate']
        self.tf_every = round(self.rate / TF_RATE)      # filter steps between transforms

        # --- reused every IMU sample, so nothing is allocated at 200 Hz ---
        self.u = np.zeros(3)                    # [a_x, a_y, r]
        self.p_diag = np.zeros(5)
        self.est = DockEstimate()
        self.fix = FiducialFix()
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = 'target'
        self.tf_msg.child_frame_id = 'base_link'

        # --- state ---
        self.zeroed = False
        self.zero_t0 = None                     # first stamp of the zeroing window
        self.zero_sum = np.zeros(3)             # sums of a_x, a_y and r while zeroing
        self.zero_count = 0
        self.accel_offset = np.zeros(2)
        self.rate_t0, self.rate_count = None, 0
        self.imu_rate = 0.0                     # measured over the last second of stamps
        self.tf_steps = 0
        self.last_accept = -math.inf            # capture time of the last accepted fix [s]
        self.last_nis = math.nan
        self.last_accepted = False

        self.est_pub = self.create_publisher(DockEstimate, 'dock/estimate', 10)
        self.fix_pub = self.create_publisher(FiducialFix, 'fiducial/fix', 10)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Imu, 'imu/data', self.on_imu, IMU_QOS)
        self.create_subscription(RosInferenceResult, 'inference_result', self.on_pose, POSE_QOS)

    def on_imu(self, msg):
        a, w = msg.linear_acceleration, msg.angular_velocity
        t = seconds(msg.header.stamp)
        self.measure_rate(t)
        if not self.zeroed:
            self.zero(msg, t, a.x, a.y, w.z)
            return
        if not self.ekf.initialised:
            return
        self.u[0] = a.x - self.accel_offset[0]
        self.u[1] = a.y - self.accel_offset[1]
        self.u[2] = w.z
        steps = self.ekf.advance(self.u, t)
        if steps:
            self.publish_estimate(msg.header.stamp, t, steps)

    def measure_rate(self, t):
        """Count IMU samples over each second of their stamps; warn when off imu.rate."""
        if self.rate_t0 is None:
            self.rate_t0 = t
            return
        self.rate_count += 1
        if t - self.rate_t0 >= 1.0:
            self.imu_rate = self.rate_count / (t - self.rate_t0)
            self.rate_t0, self.rate_count = t, 0
            if abs(self.imu_rate - self.rate) > RATE_TOL * self.rate:
                self.get_logger().warning(
                    f'IMU at {self.imu_rate:.0f} Hz, configured for {self.rate:.0f} Hz: the '
                    f'EKF still steps T = {1e3 / self.rate:.0f} ms, but the supervisor will '
                    'not start moving', throttle_duration_sec=5.0)

    def zero(self, msg, t, ax, ay, r):
        """
        Average the stationary start-up window into the accelerometer offset (brief Section 5).

        The design treats the residual bias as noise, not a state. The robot must
        be still while the node starts. The gyro mean is logged for the bias check
        of brief Section 10.a (step 4), not subtracted.
        """
        if self.zero_t0 is None:
            self.zero_t0 = t
            if msg.linear_acceleration_covariance[0] == -1.0:
                self.get_logger().error(
                    'imu/data carries no linear acceleration (covariance[0] = -1); the EKF '
                    'is driven by it, so the IMU node has to publish it')
        self.zero_sum[0] += ax
        self.zero_sum[1] += ay
        self.zero_sum[2] += r
        self.zero_count += 1
        if t - self.zero_t0 >= ZERO_S:
            mean = self.zero_sum / self.zero_count
            self.accel_offset[:] = mean[:2]
            self.zeroed = True
            self.get_logger().info(
                f'IMU zeroed over {ZERO_S:.1f} s: accel offset ({mean[0]:+.4f}, {mean[1]:+.4f}) '
                f'm/s^2, gyro mean {math.degrees(mean[2]):+.3f} deg/s')

    def on_pose(self, msg):
        if msg.reason or not self.zeroed:
            return                              # refused by the pipeline, or still zeroing
        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.exposure_ns
        stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        t_capture = ns * 1e-9
        fix, cov_valid = self.measure(msg.pose), msg.covariance_valid
        if msg.ambiguous:
            # Two IPPE solutions (docking_design.md 2.3): keep the one nearer the
            # prediction, and never start the filter from an ambiguous pose.
            if not self.ekf.initialised:
                return
            alt = self.measure(msg.pose_alt)
            if alt is not None and (fix is None or self.ekf.nis(alt[0], alt[1], t_capture)
                                    < self.ekf.nis(fix[0], fix[1], t_capture)):
                fix, cov_valid = alt, msg.covariance_valid_alt
        if fix is None:
            self.get_logger().warning('a pose arrived without a usable covariance: not fused',
                                      throttle_duration_sec=5.0)
            return
        z, R = fix[0], fix[1]
        self.publish_fix(stamp, fix, cov_valid, msg.ambiguous)

        if not self.ekf.initialised:
            self.ekf.initialise(z, R, t_capture)
            self.last_accept, self.last_accepted = t_capture, True
            return
        self.last_accepted, self.last_nis = self.ekf.update(z, R, t_capture)
        if self.last_accepted:
            self.last_accept = t_capture

    def measure(self, pose):
        """
        Run the chain on one PoseWithCovariance; return (z, R, roll, pitch, range).

        R comes from the pose's own covariance, measured live by the pipeline for
        this frame (FiducialChain.measure). None if that covariance is unusable:
        a pose is never fused with a made-up noise.
        """
        cov = np.array(pose.covariance).reshape(6, 6)
        if not (np.all(np.isfinite(cov)) and np.all(cov.diagonal() > 0.0)):
            return None
        q, t = pose.pose.orientation, pose.pose.position
        t_OM = np.array((t.x, t.y, t.z))
        z, R, roll, pitch = self.chain.measure(quat_to_rot(q.x, q.y, q.z, q.w), t_OM, cov)
        return z, R, roll, pitch, math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)

    def publish_estimate(self, stamp, t, steps):
        m, ekf = self.est, self.ekf
        np.copyto(self.p_diag, ekf.P.diagonal())
        m.header.stamp = stamp
        m.x = ekf.x
        m.p_diag = self.p_diag
        m.r_gyro = float(self.u[2])
        m.imu_rate = self.imu_rate
        m.fix_age = t - self.last_accept
        m.last_nis = float(self.last_nis)
        m.last_fix_accepted = bool(self.last_accepted)
        self.est_pub.publish(m)
        self.tf_steps += steps                  # 20 Hz of filter time, whatever the IMU rate
        if self.tf_steps >= self.tf_every:
            self.tf_steps %= self.tf_every
            self.publish_tf(stamp)

    def publish_fix(self, stamp, fix, cov_valid, ambiguous):
        (z, R, roll, pitch, d), m = fix, self.fix
        m.header.stamp = stamp
        m.x, m.y, m.theta = float(z[0]), float(z[1]), float(z[2])
        m.covariance = R.ravel()
        m.covariance_valid = bool(cov_valid)
        m.roll, m.pitch = float(roll), float(pitch)
        m.range = d
        m.ambiguous = bool(ambiguous)
        self.fix_pub.publish(m)

    def publish_tf(self, stamp):
        m, x = self.tf_msg, self.ekf.x
        m.header.stamp = stamp
        m.transform.translation.x = float(x[0])
        m.transform.translation.y = float(x[1])
        m.transform.rotation.z = math.sin(0.5 * x[2])
        m.transform.rotation.w = math.cos(0.5 * x[2])
        self.tf_pub.sendTransform(m)


def main(args=None):
    run(EkfNode, args=args)


if __name__ == '__main__':
    main()
