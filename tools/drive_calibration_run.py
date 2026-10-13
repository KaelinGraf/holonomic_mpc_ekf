#!/usr/bin/env python3
"""
Step the drive one axis and one speed at a time, with the board in view, and bag what it did.

    python3 tools/drive_calibration_run.py --dry-run     # check the inputs and print the plan
    python3 tools/drive_calibration_run.py               # the full run, about two minutes
    python3 tools/drive_calibration_run.py --axes y      # strafe only

On the Pi, in the container, with pi.launch.py up (the bridge, the camera, the IMU and
dock_control, whose ekf_node publishes fiducial/fix for every board pose, armed or not) and the
laptop's inference pipeline solving poses. Then tools/drive_calibration_fit.py <bag>, on the
laptop, fits each axis's deadband and gain from the bag.

Set the robot on the floor about 1 m in front of the board, facing it, with room to move 0.5 m
every way. The run arms the drive alone (/serial_bridge/arm), never the docking (dock/arm), so the
MPC stays off cmd_vel; it will not start unless docking is idle, the drive link is up and fixes
are arriving. Each step holds one axis at one speed for --hold s, then --rest s of zeros, with
forward and back (left and right, anticlockwise and clockwise) in turn, so the robot ends near
where it began:

    x (forward +, back -)          0.02 0.04 0.06 0.10 0.15 m/s
    y (left +, right -)            0.02 0.04 0.06 0.10 0.15 m/s
    r (anticlockwise +, cw -)      0.05 0.1  0.2  0.3  0.5  rad/s

Before each forward or sideways step the board must be in view (a fix under --max-fix-age old; the
turns are measured by the gyro, so they go ahead without it); after each step, a line says how far
the fixes say it went. The bag (bags/drivecal_<time>/, with the run's plan in steps.json)
holds cmd_vel, drive/telemetry, drive/status, fiducial/fix, imu/data, inference_result and
dock/stage. Ctrl-C, a lost drive link, a disarm from elsewhere or docking being armed stop the run:
zeros, disarm, and the bag is closed with what it has (Ctrl-C reaches the recorder too, so it
closes the bag before the disarm).

Light enough for the Pi's shared core: it pins itself (and the recorder it starts) to --core, 0 by
default, and subscribes to nothing faster than the 15 Hz fixes.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

AXES = {'x': 'forward', 'y': 'left', 'r': 'anticlockwise'}
UNITS = {'x': 'm/s', 'y': 'm/s', 'r': 'rad/s'}
SPEEDS = {'x': (0.02, 0.04, 0.06, 0.10, 0.15), 'y': (0.02, 0.04, 0.06, 0.10, 0.15),
          'r': (0.05, 0.1, 0.2, 0.3, 0.5)}
TOPICS = ('/cmd_vel', '/drive/telemetry', '/drive/status', '/fiducial/fix', '/imu/data',
          '/inference_result', '/dock/stage')
COMMAND_HZ = 50.0           # cmd_vel rate; the bridge sends zeros after its cmd_timeout (0.1 s)
SETTLE_S = 0.4              # left out of each step's live summary, while the motors spin up
STOP_TIMEOUT_S = 10.0       # for the recorder to close its bag after SIGINT


class Abort(Exception):
    """Stop the run: zeros, disarm, close the bag."""


def plan(axes):
    """Return the steps, in order: (axis, signed command)."""
    steps = []
    for axis in axes:
        for speed in SPEEDS[axis]:
            steps += [(axis, speed), (axis, -speed)]
    return steps


def twist_of(axis, value):
    """(vx, vy, wz) with one axis set."""
    return (value if axis == 'x' else 0.0, value if axis == 'y' else 0.0,
            value if axis == 'r' else 0.0)


def summarise(fixes, t0, t1):
    """Return the fixes' (n, span, forward, left, turn [rad]) over [t0, t1], in the body frame."""
    use = [f for f in fixes if t0 <= f[0] <= t1]
    clear = [f for f in use if not f[4]]
    use = clear if len(clear) >= 5 else use
    if len(use) < 3:
        return len(use), 0.0, math.nan, math.nan, math.nan
    a = np.array([f[:4] for f in use])
    t, x, y = a[:, 0] - a[0, 0], a[:, 1], a[:, 2]
    th = np.unwrap(a[:, 3])
    span = t[-1]
    vx, vy = np.polyfit(t, x, 1)[0], np.polyfit(t, y, 1)[0]     # least-squares slopes, world
    c, s = math.cos(float(np.median(th))), math.sin(float(np.median(th)))
    fwd, left = (c * vx + s * vy) * span, (-s * vx + c * vy) * span
    return len(use), span, fwd, left, float(th[-1] - th[0])


class Run:
    """The ROS side: one node, spun by hand between publishes."""

    def __init__(self, rclpy, args):
        from convchart_interfaces.msg import DriveStatus
        from dock_msgs.msg import DockStage, FiducialFix
        from geometry_msgs.msg import Twist
        from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                               QoSReliabilityPolicy)
        from std_srvs.srv import SetBool

        self.rclpy, self.args = rclpy, args
        self.DriveStatus, self.DockStage, self.Twist, self.SetBool = (DriveStatus, DockStage,
                                                                      Twist, SetBool)
        self.node = rclpy.create_node('drive_calibration_run')
        # As mpc_node: the bridge subscribes RELIABLE, depth 1, and keeps only the newest.
        cmd_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        # drive/status is latched (TRANSIENT_LOCAL): the current state arrives on connection.
        status_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                                reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', cmd_qos)
        self.arm_client = self.node.create_client(SetBool, '/serial_bridge/arm')
        self.fixes = []             # (capture stamp, x, y, theta, ambiguous, range), and when
        self.fix_arrivals = []
        self.status = None
        self.status_arrival = 0.0
        self.stage = None
        self.armed_here = False     # the run sent an arm: it must disarm on the way out
        self.watching = False       # the arm took: check() runs at every publish
        self.node.create_subscription(FiducialFix, '/fiducial/fix', self.on_fix, 10)
        self.node.create_subscription(DriveStatus, '/drive/status', self.on_status, status_qos)
        self.node.create_subscription(DockStage, '/dock/stage', self.on_stage, 10)

    # --- inputs ---------------------------------------------------------------

    def on_fix(self, m):
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.fixes.append((st, m.x, m.y, m.theta, m.ambiguous, m.range))
        self.fix_arrivals.append(time.monotonic())
        del self.fixes[:-2000], self.fix_arrivals[:-2000]

    def on_status(self, m):
        self.status = m
        self.status_arrival = time.monotonic()

    def on_stage(self, m):
        self.stage = m.stage

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def fix_age(self):
        return time.monotonic() - self.fix_arrivals[-1] if self.fix_arrivals else math.inf

    def spin_for(self, seconds, command=None):
        """Spin for seconds; with a command, publish it at COMMAND_HZ and check it is safe to."""
        msg = None
        if command is not None:
            msg = self.Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = command
        end = time.monotonic() + seconds
        next_pub = time.monotonic()
        while True:
            now = time.monotonic()
            if msg is not None and now >= next_pub:
                self.cmd_pub.publish(msg)
                next_pub = max(next_pub + 1.0 / COMMAND_HZ, now)
                if self.watching:
                    self.check()
            if now >= end:
                return
            wake = min(end, next_pub) if msg is not None else end
            self.rclpy.spin_once(self.node, timeout_sec=max(0.0, min(wake - now, 0.05)))

    def check(self):
        """Abort if docking was armed, or the drive disarmed or lost, while the run had it."""
        if self.stage is not None and self.stage != self.DockStage.IDLE:
            raise Abort(f'docking left IDLE (stage {self.stage}): the MPC would fight this run')
        if self.status.state < self.DriveStatus.STALLED:
            raise Abort(f'the drive is no longer armed: {self.status.detail}')
        if time.monotonic() - self.status_arrival > 2.0:            # it beats at 2 Hz
            raise Abort('drive/status has gone quiet: is the serial bridge still running?')

    # --- arming ---------------------------------------------------------------

    def arm(self, on):
        """Call /serial_bridge/arm; True when the bridge says it went out."""
        if not self.arm_client.wait_for_service(timeout_sec=2.0):
            return False
        future = self.arm_client.call_async(self.SetBool.Request(data=on))
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
        ok = future.done() and future.result() is not None and future.result().success
        print(f'  {"arm" if on else "disarm"}: '
              f'{future.result().message if future.done() and future.result() else "no reply"}',
              flush=True)
        return ok


def preflight(run, args):
    """Wait for the inputs; return a reason not to start, or ''."""
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        run.spin_for(0.1)
        recent = sum(1 for t in run.fix_arrivals if time.monotonic() - t < 1.0)
        if run.status is not None and run.stage is not None and recent >= args.min_fix_rate:
            break
    S = run.DriveStatus
    if run.status is None:
        return 'no drive/status: is pi.launch.py (the serial bridge) running?'
    if run.status.state < S.DISARMED:
        return f'the drive link is down: {run.status.detail}'
    if run.stage is None:
        return 'no dock/stage: is dock_control running? (its ekf_node publishes the fixes)'
    if run.stage != run.DockStage.IDLE:
        return f'docking is armed (stage {run.stage}): disarm it first, or the MPC fights the run'
    recent = sum(1 for t in run.fix_arrivals if time.monotonic() - t < 1.0)
    if recent < args.min_fix_rate:
        return (f'{recent} fiducial/fix in the last second (need {args.min_fix_rate}): put the '
                f'board in view, and check the laptop is solving poses')
    return ''


def start_recorder(path):
    """Start ros2 bag record of TOPICS into path, logging to path.log."""
    log = open(path + '.log', 'w')
    proc = subprocess.Popen(['ros2', 'bag', 'record', '-o', path, *TOPICS], stdout=log,
                            stderr=subprocess.STDOUT,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    return proc, log


def recorder_subscribed(path):
    """Return the TOPICS the recorder's log says it has subscribed to."""
    try:
        text = open(path + '.log').read()
    except OSError:
        return set()
    return {t for t in TOPICS if f"Subscribed to topic '{t}'" in text}


def stop_recorder(proc):
    """SIGINT the recorder, which closes the bag with its metadata, and wait for it."""
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(STOP_TIMEOUT_S)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0],
                                 epilog='See the module docstring for the setup.')
    ap.add_argument('--axes', default='xyr', help='which of x, y, r to step, in order')
    ap.add_argument('--hold', type=float, default=1.5, help='seconds each command is held')
    ap.add_argument('--rest', type=float, default=2.0, help='seconds of zeros after each')
    ap.add_argument('--bag-dir', default='bags', help='where the bag goes (default ./bags)')
    ap.add_argument('--core', type=int, default=0, help='CPU to pin to; -1 leaves it unpinned')
    ap.add_argument('--wait', type=float, default=5.0, help='seconds to wait for the inputs')
    ap.add_argument('--min-fix-rate', type=int, default=5, help='fixes a second needed to start')
    ap.add_argument('--max-fix-age', type=float, default=1.0,
                    help='x and y steps wait (in zeros, up to 5 s) for a fix this fresh')
    ap.add_argument('--dry-run', action='store_true', help='check and print the plan; no motion')
    args, ros_args = ap.parse_known_args(argv)
    axes = [a for a in args.axes if a in AXES]
    if not axes or len(axes) != len(args.axes):
        ap.error('--axes takes x, y and r, e.g. --axes xy')
    steps = plan(axes)
    total = len(steps) * (args.hold + args.rest)
    if args.core >= 0:
        os.sched_setaffinity(0, {args.core})       # the recorder inherits it

    import rclpy
    from rclpy.signals import SignalHandlerOptions
    # Our own SIGINT handling: Ctrl-C leaves the context alive to send the zeros and the disarm.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)           # the ssh session went away: stop, still

    run = Run(rclpy, args)
    proc = log = None
    path = ''
    done = []
    reason = ''
    try:
        print(f'drive calibration: {len(steps)} steps on {"".join(axes)}, {args.hold:g} s each '
              f'then {args.rest:g} s still: about {total:.0f} s', flush=True)
        why = preflight(run, args)
        if why:
            sys.exit(f'not starting: {why}')
        last = run.fixes[-1]
        amb = np.mean([f[4] for f in run.fixes[-15:]])
        print(f'board in view: range {last[5]:.2f} m, base at x {last[1]:+.3f} y {last[2]:+.3f} '
              f'm, theta {math.degrees(last[3]):+.1f} deg; {amb:.0%} of recent fixes ambiguous; '
              f'drive {run.status.detail}', flush=True)
        for i, (axis, value) in enumerate(steps, 1):
            print(f'  step {i:2d}: {axis} {value:+.3f} {UNITS[axis]}', flush=True)
        if args.dry_run:
            print('dry run: nothing armed, nothing recorded')
            return

        os.makedirs(args.bag_dir, exist_ok=True)
        path = os.path.join(os.path.abspath(args.bag_dir),
                            time.strftime('drivecal_%Y%m%d_%H%M%S'))
        proc, log = start_recorder(path)
        deadline = time.monotonic() + 10.0
        while recorder_subscribed(path) != set(TOPICS) and time.monotonic() < deadline:
            run.spin_for(0.2)
            if proc.poll() is not None:
                raise Abort(f'the recorder exited ({proc.returncode}); see {path}.log')
        missing = set(TOPICS) - recorder_subscribed(path)
        if missing:
            print(f'  the recorder has not seen {", ".join(sorted(missing))}; going on',
                  flush=True)
        print(f'recording {path}; 2 s still first', flush=True)
        run.spin_for(2.0)

        run.armed_here = True
        if not run.arm(True):
            raise Abort('the bridge did not arm the drive')
        deadline = time.monotonic() + 3.0
        while run.status.state < run.DriveStatus.STALLED:
            if time.monotonic() > deadline:
                raise Abort(f'the drive did not report armed: {run.status.detail}')
            run.spin_for(0.05, (0.0, 0.0, 0.0))
        run.watching = True
        run.spin_for(1.0, (0.0, 0.0, 0.0))

        for i, (axis, value) in enumerate(steps, 1):
            deadline = time.monotonic() + 5.0
            while axis != 'r' and run.fix_age() > args.max_fix_age:
                if time.monotonic() > deadline:
                    raise Abort(f'no fix for {run.fix_age():.1f} s: the board is out of view, '
                                f'or the laptop stopped solving poses')
                run.spin_for(0.1, (0.0, 0.0, 0.0))
            print(f'step {i:2d}/{len(steps)}: {AXES[axis]} {value:+.3f} {UNITS[axis]}', end='',
                  flush=True)
            t0 = run.now()
            run.spin_for(args.hold, twist_of(axis, value))
            t1 = run.now()
            run.spin_for(args.rest, (0.0, 0.0, 0.0))
            done.append({'axis': axis, 'cmd': value, 't_start': t0, 't_end': t1})
            n, span, fwd, left, turn = summarise(run.fixes, t0 + SETTLE_S, t1)
            if n < 3:
                print(f'   {n} fixes in the step: the board was not seen', flush=True)
            else:
                print(f'   moved {fwd * 100:+5.1f} cm fwd {left * 100:+5.1f} cm left '
                      f'{math.degrees(turn):+6.1f} deg in {span:.1f} s ({n} fixes, range '
                      f'{run.fixes[-1][5]:.2f} m)', flush=True)
        run.spin_for(1.0, (0.0, 0.0, 0.0))
    except Abort as e:
        reason = str(e)
        print(f'\nstopping: {reason}', flush=True)
    except KeyboardInterrupt:
        reason = 'interrupted'
        print('\nstopping: interrupted', flush=True)
    finally:
        run.watching = False
        if run.armed_here:
            run.spin_for(0.2, (0.0, 0.0, 0.0))
            if not run.arm(False):
                print('  !! the disarm did not go through: call '
                      "ros2 service call /serial_bridge/arm std_srvs/srv/SetBool '{data: false}'",
                      flush=True)
            run.spin_for(1.0)
        stop_recorder(proc)
        if log is not None:
            log.close()
        if path and os.path.isdir(path):
            with open(os.path.join(path, 'steps.json'), 'w') as f:
                json.dump({'hold_s': args.hold, 'rest_s': args.rest, 'axes': ''.join(axes),
                           'steps': done, 'planned': len(steps), 'stopped': reason or None}, f,
                          indent=1)
            print(f'bag: {path} ({len(done)} of {len(steps)} steps). Copy it to the laptop and '
                  f'fit it there: python3 tools/drive_calibration_fit.py <bag dir>', flush=True)
        run.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
