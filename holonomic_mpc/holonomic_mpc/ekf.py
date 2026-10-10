import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class EKF(Node):

    def __init__(self):
        super().__init__('ekf')


def main(args=None):
    rclpy.init(args=args)
    node = EKF()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
