"""ROS plumbing shared by the nodes: spinning several nodes in one process, stamps."""

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor


def run(*node_types, args=None):
    """Construct node_types and spin them in one single-threaded executor until shutdown.

    Single-threaded so that the EKF's IMU and fix callbacks never overlap.

    Args:
        *node_types: Node classes, each constructed without arguments.
        args: Command-line arguments for rclpy.init.
    """
    rclpy.init(args=args)
    executor = SingleThreadedExecutor()
    nodes = []
    try:
        for node_type in node_types:
            nodes.append(node_type())
            executor.add_node(nodes[-1])
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.try_shutdown()


def seconds(stamp):
    """Convert a builtin_interfaces/Time to float seconds.

    Args:
        stamp: The time stamp.
    """
    return stamp.sec + stamp.nanosec * 1e-9
