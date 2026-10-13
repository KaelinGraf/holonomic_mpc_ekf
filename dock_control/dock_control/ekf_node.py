"""
EKF node (brief Section 5): IMU and board poses in, the docking estimate out.

Inputs come from the existing nodes: imu/data from p4p_imu at the IMU rate, and
inference_result from the convchart_ros pipeline, the board pose in the camera
frame stamped with the Pi's image stamp. The fiducial chain runs inside the pose
callback, so a pose costs no extra node or topic hop. One single-threaded
executor, so the IMU and pose callbacks never overlap.

Publishes dock/estimate after every IMU sample, fiducial/fix for every pose (the
log and the supervisor's roll and pitch monitor), and target -> base_link on TF
at 20 Hz.

The filter runs only while docking is armed (dock/stage out of IDLE). Arming
restarts it: ZERO_S of stationary IMU zeroing, then the first accepted pose
initialises it. Disarming idles it, so no drift builds up between runs.
fiducial/fix goes out for every pose either way.

An ambiguous pose (two IPPE solutions, docking_design.md 2.3) is not refused.
Once the filter runs, the solution with the smaller NIS against it is fused, and
Ekf.update's gate still rejects it if even that one is too far (resolve()). The
filter starts from the first solution (the smaller reprojection error) of the
first pose after zeroing, ambiguous or not, and corrects itself with the poses
that follow (the user's decision, 2026-10-13: until then it waited for the two
solutions to agree, which at range could take ten seconds). The fused
solution's noise has the two solutions' disagreement added to it: the pipeline's
covariance is that of one solution, and on the robot (2026-10-13) an ambiguous
pose's heading stood still while the gyro saw the robot turning. That noise also
starts the filter wide where the two disagree.

While docking is in ACQUIRE, poses are fused without the NIS gate (the user's
decision, 2026-10-13): the filter has only just started from what may be the
wrong one of two solutions, and must be free to move to the right one.

If REANCHOR_AFTER fixes in a row fail the gate yet agree with each other, it is
the filter that has gone wrong (a bump at the dock, a drift through a run of
rejected fixes), and the filter restarts from the newest of them rather than
drifting on until the covariance aborts the run.

While the robot stands, the filter knows its velocity: once cmd_vel has been zero
(or absent) for ekf.zupt_settle_s, every step ends with the velocity reset to zero
(Ekf.zero_velocity), until a moving command arrives. Without it, a board moved by
hand looked like the robot moving, and the velocity stayed when the board left
the view (2026-10-13).

The filter steps each sample's own interval (Ekf.advance), so it stays right
whatever rate the IMU delivers and if it drops samples; the measured IMU rate
goes out in every estimate, and the supervisor will not start moving below 90 %
of imu.rate, the sign of a dead or failing IMU.

imu/data carries acceleration including gravity (sensor_msgs/Imu; the BNO085's
calibrated accelerometer report), and the EKF wants it removed (report Section
2.1). Each sample's own orientation gives gravity in the body frame, from roll
and pitch alone, which is subtracted before the zeroing and the predict.
"""

from collections import deque
import math

from builtin_interfaces.msg import Time
from convchart_interfaces.msg import RosInferenceResult
from convchart_qos.qos import INFERENCE_RESULT_QOS
from dock_control import params
from dock_control.ekf import Ekf
from dock_control.fiducial import FiducialChain
from dock_control.geometry import body_gravity_xy, quat_to_rot, wrap_pi
from dock_control.rosutil import run, seconds
from dock_msgs.msg import DockEstimate, DockStage, FiducialFix
from geometry_msgs.msg import TransformStamped, Twist
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

TF_RATE = 20.0      # Hz, target -> base_link (brief Section 3)
ZERO_S = 2.0        # s of stationary IMU averaged into the accelerometer offset (brief Section 5)
RATE_TOL = 0.1      # warn when the measured IMU rate is below the configured one by more
AGREE_SIGMAS = 3.0  # planar_agree: an ambiguous pose's solutions agree to this (diagnostic)
# Re-anchoring: this many fixes in a row failing the gate but agreeing with each other (each
# within REANCHOR_P and REANCHOR_TH_DEG of the one before) mean the filter, not the camera,
# has gone wrong; the filter restarts from the newest of them instead of drifting on.
REANCHOR_AFTER = 5
REANCHOR_P = 0.03
REANCHOR_TH_DEG = 3.0

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
# The MPC's command, as the bridge reads it (mpc_node.CMD_QOS).
CMD_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE)
MOVING_CMD = 1e-4   # a command component above this [m/s, rad/s] moves the robot


class EkfNode(Node):

    def __init__(self):
        super().__init__('ekf_node')
        p = params.from_node(self)
        # Image stamp minus capture time: the Pi stamps a frame when it reads it,
        # after exposure and readout. Measure it once with the LED test (brief
        # Section 10.b, step 3).
        exposure_offset = self.declare_parameter('exposure_offset', 0.0).value

        # --- constant, built once ---
        self.p = p
        self.ekf = Ekf(p)
        self.chain = FiducialChain(p)
        self.exposure_ns = round(exposure_offset * 1e9)
        self.rate = p['imu']['rate']
        # Once the commands have been zero (or absent) this long, the robot has
        # stopped: the EKF holds its velocity at zero (Ekf.zero_velocity).
        self.settle_s = p['ekf'].get('zupt_settle_s', 0.3)

        # --- reused every IMU sample, so nothing is allocated at 150 Hz ---
        self.u = np.zeros(3)                    # [a_x, a_y, r]
        self.p_diag = np.zeros(5)
        self.est = DockEstimate()
        self.fix = FiducialFix()
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = 'target'
        self.tf_msg.child_frame_id = 'base_link'

        # --- state ---
        self.active = False                     # docking armed: the filter runs
        self.acquiring = False                  # dock/stage is ACQUIRE: fuse without the gate
        self.zeroed = False
        self.zero_t0 = None                     # first stamp of the zeroing window
        self.zero_sum = np.zeros(3)             # sums of a_x, a_y and r while zeroing
        self.zero_count = 0
        self.accel_offset = np.zeros(2)
        self.rate_t0, self.rate_count = None, 0
        self.imu_rate = 0.0                     # measured over the last second of stamps
        self.tf_window = -1                     # the last 1/TF_RATE window that got a transform
        self.last_accept = -math.inf            # capture time of the last accepted fix [s]
        self.last_nis = math.nan
        self.last_accepted = False
        self.last_motion = -math.inf            # when the last moving command arrived [s]
        self.rejected = deque(maxlen=REANCHOR_AFTER)    # the fixes the gate turned away in a row
        self.reanchors = 0
        self.stopped = True                     # as last reported in the log

        self.est_pub = self.create_publisher(DockEstimate, 'dock/estimate', 10)
        self.fix_pub = self.create_publisher(FiducialFix, 'fiducial/fix', 10)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Imu, 'imu/data', self.on_imu, IMU_QOS)
        self.create_subscription(RosInferenceResult, 'inference_result', self.on_pose, POSE_QOS)
        self.create_subscription(DockStage, 'dock/stage', self.on_stage, 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, CMD_QOS)

    def on_imu(self, msg):
        a, w, o = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        t = seconds(msg.header.stamp)
        self.measure_rate(t)
        if not self.active:
            return
        gx, gy = body_gravity_xy(o.x, o.y, o.z, o.w)    # gravity at this sample's roll, pitch
        ax, ay = a.x - gx, a.y - gy
        if not self.zeroed:
            self.zero(msg, t, ax, ay, w.z)
            return
        if not self.ekf.initialised:
            return
        self.u[0] = ax - self.accel_offset[0]
        self.u[1] = ay - self.accel_offset[1]
        self.u[2] = w.z
        stopped = t - self.last_motion >= self.settle_s
        if stopped != self.stopped:
            self.stopped = stopped
            self.get_logger().info('commands stopped: velocity held at zero' if stopped
                                   else 'commands moving: velocity estimated')
        if self.ekf.advance(self.u, t, stopped):
            self.publish_estimate(msg.header.stamp, t)

    def on_cmd(self, msg):
        """Note when the robot was last told to move; zero (or no) commands mean it stands."""
        if max(abs(msg.linear.x), abs(msg.linear.y), abs(msg.angular.z)) > MOVING_CMD:
            self.last_motion = self.get_clock().now().nanoseconds * 1e-9

    def on_stage(self, msg):
        """Estimate only while docking is armed: leaving IDLE restarts the filter."""
        active = msg.stage != DockStage.IDLE
        if active and not self.active:
            self.restart()
            self.get_logger().info(
                f'docking armed: zeroing the IMU over {ZERO_S:.1f} s (keep the robot still), '
                'then the next pose starts the filter')
        elif self.active and not active:
            self.get_logger().info('docking disarmed: the EKF is idle until the next arm')
        self.active = active
        self.acquiring = msg.stage == DockStage.ACQUIRE

    def restart(self):
        """Forget the estimate and the IMU offset: zero again, then initialise from a pose."""
        self.ekf = Ekf(self.p)
        self.zeroed, self.zero_t0, self.zero_count = False, None, 0
        self.zero_sum[:] = 0.0
        self.last_accept, self.last_nis, self.last_accepted = -math.inf, math.nan, False
        self.tf_window = -1
        self.last_motion, self.stopped = -math.inf, True
        self.rejected.clear()

    def measure_rate(self, t):
        """Count IMU samples over each second of their stamps; warn when below imu.rate."""
        if self.rate_t0 is None:
            self.rate_t0 = t
            return
        self.rate_count += 1
        if t - self.rate_t0 >= 1.0:
            self.imu_rate = self.rate_count / (t - self.rate_t0)
            self.rate_t0, self.rate_count = t, 0
            if self.imu_rate < (1.0 - RATE_TOL) * self.rate:
                self.get_logger().warning(
                    f'IMU at {self.imu_rate:.0f} Hz, below {(1.0 - RATE_TOL) * self.rate:.0f} Hz '
                    f'({1.0 - RATE_TOL:.0%} of imu.rate): the EKF still steps each sample, '
                    'but the supervisor will not start moving', throttle_duration_sec=5.0)

    def zero(self, msg, t, ax, ay, r):
        """
        Average the stationary start-up window into the accelerometer offset (brief Section 5).

        The design treats the residual bias as noise, not a state. The robot must
        be still for ZERO_S after docking is armed. ax and ay already have gravity
        removed, so what is averaged here is the calibration residual. The gyro mean is logged
        for the bias check of brief Section 10.a (step 4), not subtracted.
        """
        if self.zero_t0 is None:
            self.zero_t0 = t
            if msg.linear_acceleration_covariance[0] == -1.0:
                self.get_logger().error(
                    'imu/data carries no linear acceleration (covariance[0] = -1); the EKF '
                    'is driven by it, so the IMU node has to publish it')
            if msg.orientation_covariance[0] == -1.0:
                self.get_logger().error(
                    'imu/data carries no orientation (covariance[0] = -1): gravity cannot be '
                    'removed as the robot tilts, only the start-up offset')
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
        if msg.reason:
            return                              # refused by the pipeline
        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec - self.exposure_ns
        stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        t_capture = ns * 1e-9
        solutions = [(msg.pose, msg.covariance_valid)]
        if msg.ambiguous:
            solutions.append((msg.pose_alt, msg.covariance_valid_alt))
        measured = ((self.measure(pose), valid) for pose, valid in solutions)
        candidates = [(fix, valid) for fix, valid in measured if fix is not None]
        if not candidates:
            self.get_logger().warning('a pose arrived without a usable covariance: not fused',
                                      throttle_duration_sec=5.0)
            return
        estimating = self.active and self.zeroed
        fix, cov_valid, may_start = resolve(self.ekf, candidates, msg.ambiguous, t_capture,
                                            estimating)
        z, R = fix[0], fix[1]
        self.publish_fix(stamp, fix, cov_valid, msg.ambiguous)
        if not estimating:
            return                              # not armed, or still zeroing: fix only logged
        if not self.ekf.initialised:
            if may_start:
                self.ekf.initialise(z, R, t_capture)
                self.last_accept, self.last_accepted = t_capture, True
            return
        self.last_accepted, self.last_nis = self.ekf.update(z, R, t_capture,
                                                            gate=not self.acquiring)
        if self.last_accepted:
            self.last_accept = t_capture
            self.rejected.clear()
        elif not math.isnan(self.last_nis):                 # gated, not refused for its age
            self.rejected.append((t_capture, z, R))
            if (len(self.rejected) == REANCHOR_AFTER
                    and fixes_agree([f[1] for f in self.rejected], REANCHOR_P,
                                    math.radians(REANCHOR_TH_DEG))):
                self.reanchor()

    def reanchor(self):
        """Restart the filter from the newest of REANCHOR_AFTER rejected fixes that agree."""
        t_capture, z, R = self.rejected[-1]
        self.get_logger().warning(
            f'{REANCHOR_AFTER} fixes in a row failed the gate but agree with each other: the '
            f'filter restarts from them (was x {self.ekf.x[0]:+.3f} y {self.ekf.x[1]:+.3f} '
            f'theta {math.degrees(self.ekf.x[2]):+.1f} deg, now {z[0]:+.3f} {z[1]:+.3f} '
            f'{math.degrees(z[2]):+.1f})')
        self.ekf.reanchor(z, R, t_capture)
        self.last_accept, self.last_accepted = t_capture, True
        self.rejected.clear()
        self.reanchors += 1

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

    def publish_estimate(self, stamp, t):
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
        # 20 Hz of filter time, whatever the IMU rate: the first estimate in each
        # 50 ms window goes out, so a gap sends one transform, not a burst.
        window = math.floor(t * TF_RATE)
        if window > self.tf_window:
            self.tf_window = window
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


def planar_agree(a, b, sigmas=AGREE_SIGMAS):
    """Whether two fixes (z, R, ...) agree on x, y and theta to within sigmas of a's own R."""
    d = b[0] - a[0]
    d[2] = wrap_pi(d[2])
    return bool(np.all(np.abs(d) <= sigmas * np.sqrt(np.diag(a[1]))))


def resolve(ekf, candidates, ambiguous, t_capture, estimating):
    """
    Pick the fix among a pose's usable solutions; return (fix, cov_valid, may_start).

    candidates: [(fix, cov_valid)], the pipeline's first solution (the smaller
    reprojection error) first, a second only when the pose is ambiguous and both
    have a usable covariance. With the filter running, the one with the smaller
    NIS at the capture stamp; Ekf.update still gates it. Otherwise the first, which
    may always initialise the filter (may_start is True): an ambiguous pose starts
    it from its first solution, with the disagreement in its noise, and the poses
    that follow correct it.
    """
    fix, cov_valid = candidates[0]
    if estimating and ekf.initialised and len(candidates) > 1:
        nis = [ekf.nis(f[0], f[1], t_capture) for f, _ in candidates]
        fix, cov_valid = candidates[int(nis[1] < nis[0])]
    may_start = True
    if len(candidates) > 1:
        # An ambiguous pose fits two boards almost equally well, which its own covariance
        # does not say: where the two solutions disagree, that disagreement is added to
        # its noise, so it nudges the estimate there instead of fighting the gyro.
        d = candidates[0][0][0] - candidates[1][0][0]
        d[2] = wrap_pi(d[2])
        fix = (fix[0], fix[1] + np.diag(d * d), *fix[2:])
    return fix, cov_valid, may_start


def fixes_agree(zs, tol_p, tol_th):
    """Return whether each fix lies within tol_p [m] and tol_th [rad] of the one before it."""
    for a, b in zip(zs, zs[1:]):
        if (abs(b[0] - a[0]) > tol_p or abs(b[1] - a[1]) > tol_p
                or abs(wrap_pi(b[2] - a[2])) > tol_th):
            return False
    return True


def main(args=None):
    run(EkfNode, args=args)


if __name__ == '__main__':
    main()
