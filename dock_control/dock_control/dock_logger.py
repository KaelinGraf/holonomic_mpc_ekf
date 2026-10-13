"""
Docking logger: one bag per docking attempt, of the small topics only.

It watches dock/stage. Leaving IDLE (START DOCKING, or dock/arm true) starts
`ros2 bag record` into bags/dock_<local time>. Returning to IDLE (STOP, CANCEL,
or the disarm after DOCKED or ABORT) stops it, which closes the bag with its
metadata. So each attempt is one complete bag, whoever armed it.

The camera stream (image, camera_info) is left out: at about 2 MB/s it filled a
gigabyte of the Pi's card in eight minutes. The poses the pipeline solved from
it are recorded (inference_result), and so is everything the estimator and the
controller saw or did: about 0.1 MB/s. A bag still open after max_duration_s is
closed, and the next stage message starts another.
"""

import os
import signal
import subprocess
import time

from dock_control.rosutil import run
from dock_msgs.msg import DockStage
from rclpy.node import Node

TOPICS = ('dock/stage', 'dock/estimate', 'dock/mpc_info', 'fiducial/fix', 'inference_result',
          'cmd_vel', 'drive/status', 'drive/telemetry', 'imu/data')
GLOBAL_TOPICS = ('/tf', '/tf_static', '/rosout')
STOP_TIMEOUT_S = 10.0       # for the recorder to close its bag after SIGINT


def _default_sigint():
    # Started from a background job, the recorder would inherit SIGINT as
    # ignored, and then never close its bag on the SIGINT that stops it.
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class DockLogger(Node):

    def __init__(self):
        super().__init__('dock_logger')
        self.bag_dir = os.path.abspath(self.declare_parameter('bag_dir', 'bags').value)
        self.max_duration_s = float(self.declare_parameter('max_duration_s', 600.0).value)
        ns = self.get_namespace().rstrip('/')
        self.topics = [f'{ns}/{t}' for t in TOPICS] + list(GLOBAL_TOPICS)

        # --- state ---
        self.proc = None            # the running recorder
        self.path = ''
        self.started = 0.0

        self.create_subscription(DockStage, 'dock/stage', self.on_stage, 10)
        self.create_timer(1.0, self.check)
        self.get_logger().info(f'one bag per docking attempt, into {self.bag_dir}')

    def on_stage(self, msg):
        armed = msg.stage != DockStage.IDLE
        if armed and self.proc is None:
            self.start()
        elif not armed and self.proc is not None:
            self.stop('docking disarmed')

    def start(self):
        os.makedirs(self.bag_dir, exist_ok=True)
        self.path = os.path.join(self.bag_dir, time.strftime('dock_%Y%m%d_%H%M%S'))
        with open(self.path + '.log', 'w') as log:
            self.proc = subprocess.Popen(['ros2', 'bag', 'record', '-o', self.path, *self.topics],
                                         stdout=log, stderr=subprocess.STDOUT,
                                         preexec_fn=_default_sigint)
        self.started = time.monotonic()
        self.get_logger().info(f'docking armed: recording {self.path}')

    def stop(self, why):
        proc, self.proc = self.proc, None
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(STOP_TIMEOUT_S)
        size = (sum(e.stat().st_size for e in os.scandir(self.path))
                if os.path.isdir(self.path) else 0)
        self.get_logger().info(f'{why}: closed {self.path} ({size / 1e6:.1f} MB)')

    def check(self):
        """Report a recorder that died, and close one that has run past max_duration_s."""
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self.get_logger().error(
                f'the recorder exited ({self.proc.returncode}); see {self.path}.log')
            self.proc = None
        elif time.monotonic() - self.started > self.max_duration_s:
            self.stop(f'{self.max_duration_s:.0f} s reached')

    def destroy_node(self):
        if self.proc is not None:
            self.stop('shutting down')
        super().destroy_node()


def main(args=None):
    run(DockLogger, args=args)


if __name__ == '__main__':
    main()
