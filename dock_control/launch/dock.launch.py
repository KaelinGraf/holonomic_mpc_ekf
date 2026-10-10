"""
Launch the docking stack on the Pi: ekf_node, mpc_node, static transforms and a bag.

    ros2 launch dock_control dock.launch.py                   # demo parameters, records a bag
    ros2 launch dock_control dock.launch.py rt:=true          # pinned and SCHED_FIFO 50 (the Pi)
    ros2 launch dock_control dock.launch.py record:=false

The sensors and the drive are the existing nodes of Conv-ChArT-Wireless-Inference,
launched on their own: p4p_camera and p4p_serial_bridge on the Pi, the
convchart_ros pipeline on the laptop.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from dock_control import params
from dock_control.fiducial import FiducialChain
from dock_control.geometry import rot_to_quat, se3_inv
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _static_tf(parent, child, T):
    """Return a static_transform_publisher of child in parent, from a 4x4 transform."""
    qx, qy, qz, qw = rot_to_quat(T[:3, :3])
    x, y, z = T[:3, 3]
    return Node(package='tf2_ros', executable='static_transform_publisher',
                name=f'tf_{child}', output='log',
                arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                           '--qx', str(qx), '--qy', str(qy), '--qz', str(qz), '--qw', str(qw),
                           '--frame-id', parent, '--child-frame-id', child])


def _setup(context):
    """Build the nodes once the launch arguments are known."""
    config = LaunchConfiguration('config').perform(context)
    rt = LaunchConfiguration('rt').perform(context).lower() == 'true'
    exposure_offset = float(LaunchConfiguration('exposure_offset').perform(context))
    # The very constants ekf_node uses, so that TF and the fiducial chain agree.
    chain = FiducialChain(params.load(config))

    def node(executable, core, **extra):
        # Soft real time on the Pi (brief Section 1): a core each, SCHED_FIFO 50.
        prefix = f'taskset -c {core} chrt -f 50' if rt else None
        return Node(package='dock_control', executable=executable, output='screen',
                    prefix=prefix, parameters=[{'config': config, **extra}])

    return [
        _static_tf('target', 'marker', chain.T_TM),
        # pi_camera is the optical frame p4p_camera stamps its images with.
        _static_tf('base_link', 'pi_camera', se3_inv(chain.T_OB)),
        node('ekf_node', LaunchConfiguration('ekf_core').perform(context),
             exposure_offset=exposure_offset),
        node('mpc_node', LaunchConfiguration('mpc_core').perform(context),
             qp_solver=LaunchConfiguration('qp_solver').perform(context)),
    ]


def generate_launch_description():
    """Declare the arguments worth overriding, then build the nodes from the parameter file."""
    config = os.path.join(get_package_share_directory('dock_control'), 'config', 'dock_demo.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config,
                              description='parameter file (brief Section 9)'),
        DeclareLaunchArgument('rt', default_value='false',
                              description='pin each node to its core and run it SCHED_FIFO 50'),
        DeclareLaunchArgument('ekf_core', default_value='2'),
        DeclareLaunchArgument('mpc_core', default_value='3'),
        DeclareLaunchArgument('exposure_offset', default_value='0.0',
                              description='image stamp minus capture time [s] (LED test)'),
        DeclareLaunchArgument('qp_solver', default_value='daqp',
                              description="the MPC's QP solver: daqp or osqp"),
        DeclareLaunchArgument('record', default_value='true',
                              description='ros2 bag record -a into bags/ (brief Section 11)'),
        OpaqueFunction(function=_setup),
        ExecuteProcess(cmd=['ros2', 'bag', 'record', '-a',
                            '-o', time.strftime('bags/dock_%Y%m%d_%H%M%S')],
                       output='log', condition=IfCondition(LaunchConfiguration('record'))),
    ])
