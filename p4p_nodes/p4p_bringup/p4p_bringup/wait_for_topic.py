"""Wait until a topic carries messages: the gate pi.launch.py sequences its stages on.

    ros2 run p4p_bringup wait_for_topic /imu/data sensor_msgs/msg/Imu [--count 20] [--timeout 30]
    ros2 run p4p_bringup wait_for_topic /drive/status convchart_interfaces/msg/DriveStatus --field state --min 2

Subscribes best-effort, which matches reliable and sensor-data publishers alike, and exits 0 once count
messages have arrived (printing their rate) or 1 after timeout seconds, saying how many came. With --field,
only messages whose field is at least --min count: a publisher that is up but reporting a fault is not live.
"""
import argparse
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('topic')
    ap.add_argument('type', help='e.g. sensor_msgs/msg/Imu')
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--timeout', type=float, default=30.0)
    ap.add_argument('--field', default='', help='count only messages whose field is at least --min')
    ap.add_argument('--min', type=float, default=0.0)
    args, ros_args = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    rclpy.init(args=ros_args)
    node = rclpy.create_node('wait_for_topic')
    arrivals, last = [], {}

    def on_message(msg):
        if args.field:
            last['value'] = getattr(msg, args.field)
            if float(last['value']) < args.min:
                return
        arrivals.append(time.monotonic())

    node.create_subscription(get_message(args.type), args.topic, on_message, qos_profile_sensor_data)
    deadline = time.monotonic() + args.timeout
    try:
        while len(arrivals) < args.count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if len(arrivals) < args.count:
        seen = f', last {args.field} {last["value"]}' if 'value' in last else ''
        sys.exit(f'wait_for_topic: {args.topic} gave {len(arrivals)} of {args.count} messages in '
                 f'{args.timeout:g} s' + (f' with {args.field} >= {args.min:g}{seen}' if args.field else ''))
    rate = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], 1e-9) if len(arrivals) > 1 else 0.0
    print(f'wait_for_topic: {args.topic} live, {len(arrivals)} messages at {rate:.1f} Hz', flush=True)
