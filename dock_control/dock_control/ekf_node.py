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
        self.tf_every = round(p['imu']['rate'] / TF_RATE)
        self.zero_n = round(ZERO_S * p['imu']['rate'])

        self.var_th = self.chain.sigma_th ** 2

        # --- reused every IMU sample, so nothing is allocated at 200 Hz ---
        self.u = np.zeros(3)                    # [a_x, a_y, r]
        self.p_diag = np.zeros(5)
        self.est = DockEstimate()
        self.fix = FiducialFix()
        self.fix.sigma_th = self.chain.sigma_th
        self.fix.marker_id = 0                  # one board
        self.fix.valid = True                   # refused frames are not published
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = 'target'
        self.tf_msg.child_frame_id = 'base_link'

        # --- state ---
        self.zero_sum = np.zeros(3)             # sums of a_x, a_y and r while zeroing
        self.zero_count = 0
        self.accel_offset = np.zeros(2)
        self.n = 0
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
        if self.zero_count < self.zero_n:
            self.zero(msg, a.x, a.y, w.z)
            return
        if not self.ekf.initialised:
            return
        self.u[0] = a.x - self.accel_offset[0]
        self.u[1] = a.y - self.accel_offset[1]
        self.u[2] = w.z
        t = seconds(msg.header.stamp)
        self.ekf.predict(self.u, t)
        self.publish_estimate(msg.header.stamp, t)

    def zero(self, msg, ax, ay, r):
        """
        Average the stationary start-up window into the accelerometer offset (brief Section 5).

        The design treats the residual bias as noise, not a state. The robot must
        be still while the node starts. The gyro mean is logged for the bias check
        of brief Section 10.a (step 4), not subtracted.
        """
        if self.zero_count == 0 and msg.linear_acceleration_covariance[0] == -1.0:
            self.get_logger().error(
                'imu/data carries no linear acceleration (covariance[0] = -1); the EKF is '
                'driven by it, so the bridge has to publish it')
        self.zero_sum[0] += ax
        self.zero_sum[1] += ay
        self.zero_sum[2] += r
        self.zero_count += 1
        if self.zero_count == self.zero_n:
            mean = self.zero_sum / self.zero_n
            self.accel_offset[:] = mean[:2]
            self.get_logger().info(
                f'IMU zeroed over {ZERO_S:.1f} s: accel offset ({mean[0]:+.4f}, {mean[1]:+.4f}) '
                f'm/s^2, gyro mean {math.degrees(mean[2]):+.3f} deg/s')

    def on_pose(self, msg):
        if msg.reason or self.zero_count < self.zero_n:
            return                              # refused by the pipeline, or still zeroing
        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.exposure_ns
        stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        t_capture = ns * 1e-9
        fix = self.measure(msg.pose.pose)
        if msg.ambiguous:
            # Two IPPE solutions (docking_design.md 2.3): keep the one nearer the
            # prediction, and never start the filter from an ambiguous pose.
            if not self.ekf.initialised:
                return
            alt = self.measure(msg.pose_alt.pose)
            if self.ekf.nis(alt[0], alt[1], t_capture) < self.ekf.nis(fix[0], fix[1], t_capture):
                fix = alt
        z, R = fix[0], fix[1]
        self.publish_fix(stamp, *fix)

        if not self.ekf.initialised:
            self.ekf.initialise(z, R, t_capture)
            self.last_accept, self.last_accepted = t_capture, True
            return
        self.last_accepted, self.last_nis = self.ekf.update(z, R, t_capture)
        if self.last_accepted:
            self.last_accept = t_capture

    def measure(self, pose):
        """Run the chain on one board pose; return (z, R, roll, pitch, range)."""
        q, t = pose.orientation, pose.position
        t_OM = np.array((t.x, t.y, t.z))
        x, y, theta, roll, pitch = self.chain.chain(quat_to_rot(q.x, q.y, q.z, q.w), t_OM)
        d = math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)
        var_p = self.chain.sigma_p(d) ** 2
        return np.array((x, y, theta)), np.diag((var_p, var_p, self.var_th)), roll, pitch, d

    def publish_estimate(self, stamp, t):
        m, ekf = self.est, self.ekf
        np.copyto(self.p_diag, ekf.P.diagonal())
        m.header.stamp = stamp
        m.x = ekf.x
        m.p_diag = self.p_diag
        m.r_gyro = float(self.u[2])
        m.fix_age = t - self.last_accept
        m.last_nis = float(self.last_nis)
        m.last_fix_accepted = bool(self.last_accepted)
        self.est_pub.publish(m)
        self.n += 1
        if self.n % self.tf_every == 0:
            self.publish_tf(stamp)

    def publish_fix(self, stamp, z, R, roll, pitch, d):
        m = self.fix
        m.header.stamp = stamp
        m.x, m.y, m.theta = float(z[0]), float(z[1]), float(z[2])
        m.sigma_p = math.sqrt(R[0, 0])
        m.roll, m.pitch = float(roll), float(pitch)
        m.range = d
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
