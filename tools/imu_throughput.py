#!/usr/bin/env python3
"""Measure what an IMU stream actually delivers: rate, jitter, gaps and latency.

    python3 tools/imu_throughput.py [--topic /imu/data] [--seconds 30] [--expect-hz 150] \
        [--qos best_effort|reliable] [--clock-offset 0.0] [--json out.json]

Subscribes to sensor_msgs/Imu for --seconds and reports: the messages received and their rate against
--expect-hz (dock_control's supervisor only moves with the IMU within 10 % of its imu.rate); the spread of arrival intervals (this machine's view) and of stamp intervals (the sensor-side
cadence the EKF integrates over); gaps longer than 1.5 expected periods, which are lost or late samples;
latency from each header stamp to its arrival; and stamps that repeat or go backwards. --clock-offset is the
publisher's clock minus this machine's in seconds, so a subscriber on another machine reports one-way
latency rather than the clock difference. best_effort is what the EKF subscribes with. Run it on the Pi for
the node's own throughput, and on the laptop for what crosses the link.
"""
import argparse
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu


def stats_ms(values) -> dict:
    v = np.asarray(values, dtype=float) * 1e3
    if not v.size:
        return {}
    return {'mean': round(float(v.mean()), 3), 'std': round(float(v.std()), 3),
            'p50': round(float(np.percentile(v, 50)), 3), 'p99': round(float(np.percentile(v, 99)), 3),
            'max': round(float(v.max()), 3)}


def summarize(arrivals, stamps, accel_cov, expect_hz, clock_offset=0.0) -> dict:
    """The report for one capture: arrival times on this clock, header stamps on the publisher's, and each
    message's linear_acceleration_covariance[0]."""
    arrivals, stamps = np.asarray(arrivals, dtype=float), np.asarray(stamps, dtype=float)
    span = arrivals[-1] - arrivals[0]
    period = 1.0 / expect_hz
    d_arrival, d_stamp = np.diff(arrivals), np.diff(stamps)
    gaps = d_stamp[d_stamp > 1.5 * period]
    latency = arrivals - stamps + clock_offset
    return {
        'seconds': round(float(span), 3), 'messages': int(len(arrivals)),
        'rate_hz': round(float((len(arrivals) - 1) / span), 2) if span > 0 else 0.0,
        'stamp_rate_hz': round(float((len(stamps) - 1) / (stamps[-1] - stamps[0])), 2)
        if stamps[-1] > stamps[0] else 0.0,
        'expect_hz': expect_hz,
        'rate_error_pct': round(float(100.0 * ((len(arrivals) - 1) / span - expect_hz) / expect_hz), 2)
        if span > 0 else -100.0,
        'arrival_interval_ms': stats_ms(d_arrival),
        'stamp_interval_ms': stats_ms(d_stamp),
        'gaps_over_1.5_periods': int(len(gaps)),
        'samples_missing_in_gaps': int(np.round(gaps / period).sum() - len(gaps)) if len(gaps) else 0,
        'longest_gap_ms': round(float(gaps.max() * 1e3), 3) if len(gaps) else 0.0,
        'stamps_not_increasing': int(np.sum(d_stamp <= 0)),
        'latency_ms': stats_ms(latency),
        'clock_offset_s': clock_offset,
        'acceleration_present': bool(len(accel_cov) and accel_cov[-1] != -1.0),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--topic', default='/imu/data')
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--expect-hz', type=float, default=150.0)
    ap.add_argument('--qos', choices=('best_effort', 'reliable'), default='best_effort')
    ap.add_argument('--clock-offset', type=float, default=0.0, help='publisher clock minus this clock [s]')
    ap.add_argument('--json', default='')
    args, ros_args = ap.parse_known_args(sys.argv[1:] if argv is None else argv)

    rclpy.init(args=ros_args)
    node = rclpy.create_node('imu_throughput')
    reliability = QoSReliabilityPolicy.BEST_EFFORT if args.qos == 'best_effort' else QoSReliabilityPolicy.RELIABLE
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1000, reliability=reliability)
    arrivals, stamps, accel_cov = [], [], []

    def on_imu(msg: Imu) -> None:
        arrivals.append(time.time())
        stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        accel_cov.append(msg.linear_acceleration_covariance[0])

    node.create_subscription(Imu, args.topic, on_imu, qos)
    first_deadline = time.monotonic() + 15.0
    try:
        while not arrivals and time.monotonic() < first_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not arrivals:
            sys.exit(f'imu_throughput: nothing on {args.topic} within 15 s')
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    report = {'topic': args.topic, 'qos': args.qos,
              **summarize(arrivals, stamps, accel_cov, args.expect_hz, args.clock_offset)}
    print(json.dumps(report, indent=1))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
