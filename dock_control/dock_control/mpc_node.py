"""
MPC node (brief Sections 6-7): the supervisor and the MPC in one 20 Hz tick.

Each tick runs Supervisor.step and then Mpc.step on the newest estimate, and sends
the command to p4p_serial_bridge on cmd_vel. The bridge resends the newest
command to the firmware on its own timer and substitutes zeros once it is older
than its cmd_timeout, so a stalled controller holds station. Publishes dock/stage
and dock/mpc_info for the log.

A process of its own: one GIL per process, so the solve never delays the EKF's
IMU callbacks.
"""

import array
import time

from convchart_interfaces.msg import DriveStatus
from dock_control import params
from dock_control.mpc import Mpc
from dock_control.rosutil import run
from dock_control.supervisor import Inputs, Supervisor
from dock_msgs.msg import DockEstimate, DockStage, FiducialFix, MpcInfo
from geometry_msgs.msg import Twist
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_srvs.srv import SetBool

SOLVE_BUDGET_MS = 25.0      # of the 50 ms period (brief Section 4)

# The bridge subscribes to cmd_vel RELIABLE, depth 1: only the newest command matters.
CMD_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE)
# The bridge latches drive/status (TRANSIENT_LOCAL), so asking for the same
# delivers the current state at start-up instead of at its next heartbeat.
STATUS_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
DRIVE_OK = (DriveStatus.STALLED, DriveStatus.RUNNING)      # armed and streaming


class MpcNode(Node):

    def __init__(self):
        super().__init__('mpc_node')
        p = params.from_node(self)
        # 'daqp' (default) or 'osqp', the brief's choice; see dock_control.mpc.
        solver = self.declare_parameter('qp_solver', 'daqp').value

        # --- constant, built once: Mpc builds its constraint rows and solver here ---
        self.sup = Supervisor(p)
        self.mpc = Mpc(p, solver)
        self.get_logger().info(f'QP solver {solver}')
        self.zero_cmd = Twist()

        # --- state ---
        self.u_prev = np.zeros(3)
        self.inp = Inputs(u_last=self.u_prev)   # the supervisor reads the last command in place
        self.est = None                         # newest DockEstimate
        self.enabled = False
        self.stage = None

        # --- reused every tick ---
        self.cmd = Twist()
        self.stage_msg = DockStage()
        self.info = MpcInfo()

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', CMD_QOS)
        self.stage_pub = self.create_publisher(DockStage, 'dock/stage', 10)
        self.info_pub = self.create_publisher(MpcInfo, 'dock/mpc_info', 10)
        # The tick wants the freshest estimate (at most 5 ms old), so take all
        # 200 Hz and keep the last; a decimated topic would add up to a period of age.
        self.create_subscription(DockEstimate, 'dock/estimate', self.on_estimate, 10)
        self.create_subscription(FiducialFix, 'fiducial/fix', self.on_fix, 10)
        self.create_subscription(DriveStatus, 'drive/status', self.on_drive_status, STATUS_QOS)
        self.create_service(SetBool, 'dock/arm', self.on_arm)
        self.create_timer(p['mpc']['Ts'], self.tick)

    def on_estimate(self, msg):
        self.est = msg

    def on_fix(self, msg):
        self.inp.roll, self.inp.pitch = msg.roll, msg.pitch

    def on_drive_status(self, msg):
        self.inp.drive_ok = msg.state in DRIVE_OK

    def on_arm(self, request, response):
        self.inp.armed = request.data
        response.success = True
        response.message = 'docking armed' if request.data else 'docking disarmed'
        return response

    def tick(self):
        inp, est = self.inp, self.est
        inp.now = self.get_clock().now().nanoseconds * 1e-9
        if est is not None:
            inp.x, inp.p_diag, inp.fix_age = est.x, est.p_diag, est.fix_age
        out = self.sup.step(inp)
        self.publish_stage(out)

        if not out.command_enabled or est is None:
            if self.enabled:
                # Stop now rather than after the bridge's cmd_timeout, and start
                # the next enabled tick from rest.
                self.cmd_pub.publish(self.zero_cmd)
                self.u_prev[:] = 0.0
                self.mpc.reset()
                self.enabled = False
            return
        self.enabled = True

        t0 = time.perf_counter()
        res = self.mpc.step(est.x, est.r_gyro, self.u_prev, out.y_ref, out.v_max, out.r_max)
        solve_ms = (time.perf_counter() - t0) * 1e3
        late = solve_ms > SOLVE_BUDGET_MS
        inp.mpc_misses = inp.mpc_misses + 1 if res.fallback or late else 0
        if late and not res.fallback:
            # Brief Section 4: a tick over its budget leaves the previous command in
            # force. A fallback still applies: decelerating beats holding speed.
            self.get_logger().warning(f'tick took {solve_ms:.1f} ms: previous command kept',
                                      throttle_duration_sec=1.0)
        else:
            self.u_prev[:] = res.u
        self.publish_command(est.header.stamp, res, solve_ms)

    def publish_stage(self, out):
        if out.stage != self.stage:
            self.get_logger().info(f'stage {out.stage.name}')
            self.stage = out.stage
        m = self.stage_msg
        m.stage = int(out.stage)
        m.y_ref = out.y_ref
        m.v_max = float(out.v_max)
        m.r_max = float(out.r_max)
        m.command_enabled = bool(out.command_enabled)
        self.stage_pub.publish(m)

    def publish_command(self, stamp, res, solve_ms):
        c, u = self.cmd, self.u_prev
        c.linear.x, c.linear.y, c.angular.z = float(u[0]), float(u[1]), float(u[2])
        self.cmd_pub.publish(c)
        m = self.info
        m.header.stamp = stamp
        m.u = u
        m.y_pred = array.array('d', np.ravel(res.y_pred))
        m.slack = float(res.slack)
        m.exit_flag = int(res.exit_flag)
        m.solve_ms = solve_ms
        m.fallback = bool(res.fallback)
        self.info_pub.publish(m)


def main(args=None):
    run(MpcNode, args=args)


if __name__ == '__main__':
    main()
