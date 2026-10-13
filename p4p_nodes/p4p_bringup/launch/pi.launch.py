"""Boot the robot's Pi side in order: the sensors, then the drive link, then the docking controller.

    ros2 launch p4p_bringup pi.launch.py                     # the whole stack; nothing is armed
    ros2 launch p4p_bringup pi.launch.py stack:=sensors      # camera and IMU only
    ros2 launch p4p_bringup pi.launch.py stack:=drive        # + the Arduino bridge
    ros2 launch p4p_bringup pi.launch.py drive:=fake         # no Arduino: tools/fake_mega.py stands in for it
    ros2 launch p4p_bringup pi.launch.py camera_backend:=mock imu_backend:=mock drive:=fake rt:=false
                                                             # bench: no camera, IMU or Mega attached

Order: the camera and the IMU start together. The bridge starts once /imu/data is streaming, and the docking
controller (dock_control: EKF, MPC, static transforms, bag) once /drive/status says the Mega is streaming
(DISARMED or better: the bridge publishes NO_LINK and NO_TELEMETRY too), so the EKF's stationary
zeroing window sees real samples and the MPC has a drive to talk to. A gate that sees nothing within
stage_timeout_s stops the whole launch and names the missing input, rather than starting what depends on it.
Nothing is armed: once the controller is up the launch prints the two service calls that arm the drive and
the docking.

drive:= says what the bridge talks to. mega (the default) is the Arduino on port. fake runs tools/fake_mega.py,
which speaks the firmware's protocol (arming, watchdog, saturation) on a pseudo-terminal, so the whole stack
runs, arms and commands with nothing moving: carry the robot by hand and watch the EKF and the MPC. none starts
no bridge at all; the EKF still runs, but the MPC's supervisor never enables commands without an armed drive.

Defaults follow holonomic_mpc_ekf's README: the IMU at the EKF's 150 Hz (sample_rate_hz), the bridge with a
100 ms command watchdog (cmd_timeout) and 50 Hz set-points (command_rate_hz), and rt: the Pi's four cores one
job each. The camera (camera_core 1), the EKF (ekf_core 2) and the MPC (mpc_core 3) each get a core of their own
at SCHED_FIFO 50, and everything else (the IMU, the bridge, the bag, the transforms, the gates) runs on
other_core (0), through the launch-prefix setting that any node without a prefix of its own takes. Inside each
node, callback groups keep its own callbacks apart (the camera's frames have a thread of their own); the cores
keep the nodes apart. Each package's own launch file is included, so its other arguments pass straight through
(camera_backend, frame_rate, mock_image; imu_backend, i2c_bus, oversample; port, auto_arm; config, record,
qp_solver, exposure_offset). The frame ids are camera_frame_id, imu_frame_id and bridge_frame_id, because all
three packages call theirs frame_id, and every include is scoped so one package's default never leaks into
another's. dock_control records its bag into ./bags, relative to where this is launched from.
"""
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler, SetLaunchConfiguration, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

STACKS = ('sensors', 'drive', 'full')
DRIVES = ('mega', 'fake', 'none')
ARM = ("ros2 service call /serial_bridge/arm std_srvs/srv/SetBool '{data: true}'",
       "ros2 service call /dock/arm std_srvs/srv/SetBool '{data: true}'")


def _include(package, launch_file, prefix=None, **launch_arguments):
    """Include a package's launch file, scoped; prefix, if given, runs its nodes under it (taskset, chrt)."""
    source = PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare(package), 'launch', launch_file]))
    pinned = [SetLaunchConfiguration('launch-prefix', prefix)] if prefix else []
    return GroupAction([*pinned, IncludeLaunchDescription(source, launch_arguments=launch_arguments.items())],
                       scoped=True)


def _gate(name, topic, msg_type, count, *condition):
    return Node(package='p4p_bringup', executable='wait_for_topic', name=f'gate_{name}', output='screen',
                arguments=[topic, msg_type, '--count', str(count),
                           '--timeout', LaunchConfiguration('stage_timeout_s'), *condition])


def _then(gate, what, actions):
    def on_exit(event, context):
        if event.returncode == 0:
            return actions
        return [LogInfo(msg=f'[p4p_bringup] {what} never came up: not starting what depends on it'),
                Shutdown(reason=f'{what} is not live')]
    return RegisterEventHandler(OnProcessExit(target_action=gate, on_exit=on_exit))


def _stack(context):
    stack = LaunchConfiguration('stack').perform(context)
    if stack not in STACKS:
        raise RuntimeError(f"stack:= must be one of {', '.join(STACKS)}, got {stack!r}")
    actions, camera_prefix = [], None
    if LaunchConfiguration('rt').perform(context).lower() in ('true', '1', 'yes'):
        # Everything below that names no core of its own runs on other_core, so the camera, the EKF and the MPC
        # (dock.launch.py pins those two) have theirs to themselves.
        other, camera = (LaunchConfiguration(c).perform(context) for c in ('other_core', 'camera_core'))
        actions.append(SetLaunchConfiguration('launch-prefix', f'taskset -c {other}'))
        camera_prefix = f'taskset -c {camera} chrt -f 50'
    if LaunchConfiguration('camera').perform(context).lower() in ('true', '1', 'yes'):
        actions.append(_include('p4p_camera', 'camera.launch.py', prefix=camera_prefix,
                                frame_id=LaunchConfiguration('camera_frame_id')))
    actions.append(_include('p4p_imu', 'imu.launch.py', frame_id=LaunchConfiguration('imu_frame_id')))
    if stack == 'sensors':
        return actions
    drive = LaunchConfiguration('drive').perform(context)
    if drive not in DRIVES:
        raise RuntimeError(f"drive:= must be one of {', '.join(DRIVES)}, got {drive!r}")
    imu_gate = _gate('imu', '/imu/data', 'sensor_msgs/msg/Imu', 20)
    controller = [_include('dock_control', 'dock.launch.py'),
                  LogInfo(msg='[p4p_bringup] nothing is armed; to arm the drive, then the docking:\n  '
                              + '\n  '.join(ARM))]
    if drive == 'none':
        after_imu = [LogInfo(msg='[p4p_bringup] drive:=none: no bridge, so the MPC will not enable commands')]
        if stack == 'full':
            after_imu += [LogInfo(msg='[p4p_bringup] the IMU is live: starting dock_control'), controller[0]]
        return actions + [imu_gate, _then(imu_gate, 'the IMU (/imu/data)', after_imu)]
    bridge_args = {'frame_id': LaunchConfiguration('bridge_frame_id')}
    if drive == 'fake':
        port = LaunchConfiguration('fake_mega_port')
        actions.append(ExecuteProcess(
            cmd=['python3', PathJoinSubstitution([FindPackageShare('p4p_bringup'), 'tools', 'fake_mega.py']),
                 '--port', port], name='fake_mega', output='screen'))
        bridge_args['port'] = port
    after_imu = [LogInfo(msg=f'[p4p_bringup] the IMU is live: starting the bridge ({drive})'),
                 _include('p4p_serial_bridge', 'bridge.launch.py', **bridge_args)]
    if stack == 'full':
        drive_gate = _gate('drive', '/drive/status', 'convchart_interfaces/msg/DriveStatus', 1,
                           '--field', 'state', '--min', '2')
        after_drive = [LogInfo(msg='[p4p_bringup] the drive link is live: starting dock_control'), *controller]
        after_imu += [drive_gate, _then(drive_gate, 'the drive link (/drive/status)', after_drive)]
    return actions + [imu_gate, _then(imu_gate, 'the IMU (/imu/data)', after_imu)]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('stack', default_value='full', choices=list(STACKS),
                              description='sensors (camera, IMU), drive (+ the Arduino bridge) or full '
                                          '(+ dock_control)'),
        DeclareLaunchArgument('camera', default_value='true', description='start p4p_camera'),
        DeclareLaunchArgument('drive', default_value='mega', choices=list(DRIVES),
                              description='what the bridge talks to: mega (the Arduino on port), fake '
                                          '(tools/fake_mega.py, nothing moves) or none (no bridge)'),
        DeclareLaunchArgument('fake_mega_port', default_value='/tmp/ttyFakeMega',
                              description="drive:=fake: the fake Mega's pseudo-terminal"),
        DeclareLaunchArgument('stage_timeout_s', default_value='30.0',
                              description="how long a stage's input may take to appear before the launch stops"),
        DeclareLaunchArgument('sample_rate_hz', default_value='150.0', description="the IMU: the EKF's model rate"),
        DeclareLaunchArgument('cmd_timeout', default_value='0.1', description="the bridge's command watchdog [s]"),
        DeclareLaunchArgument('command_rate_hz', default_value='50.0',
                              description='how often the bridge resends the newest set-point'),
        DeclareLaunchArgument('rt', default_value='true',
                              description='a core each for the camera, the EKF and the MPC at SCHED_FIFO 50, '
                                          'everything else on other_core'),
        DeclareLaunchArgument('camera_core', default_value='1', description="rt: the camera's core"),
        DeclareLaunchArgument('other_core', default_value='0',
                              description='rt: the core for everything but the camera, the EKF (ekf_core) and the '
                                          'MPC (mpc_core)'),
        DeclareLaunchArgument('camera_frame_id', default_value='pi_camera', description="p4p_camera's frame_id"),
        DeclareLaunchArgument('imu_frame_id', default_value='base_link', description="p4p_imu's frame_id"),
        DeclareLaunchArgument('bridge_frame_id', default_value='base_link',
                              description="p4p_serial_bridge's frame_id"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=_stack)])
