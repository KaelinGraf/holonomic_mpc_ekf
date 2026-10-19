"""
pi.launch.py end to end on the bench: the mock camera and IMU, tools/fake_mega.py for the Arduino, and
dock_control. Each stack starts what it should and nothing more, every stage waits for its input, and a
missing input stops the launch instead of starting what depends on it.

    python3 -m pytest p4p_nodes/p4p_bringup/test/test_pi_launch.py -q

Runs `ros2 launch` on an isolated domain with discovery limited to localhost. Needs this workspace built
over the inference workspace, and the ros2 CLI; skipped otherwise.
"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

DOMAIN_ID = 7 + os.getpid() % 20
TIMEOUT_S = 40.0
FAKE_MEGA = Path(__file__).resolve().parents[3] / 'tools' / 'fake_mega.py'
BENCH = ['camera_backend:=mock', 'imu_backend:=mock', 'rt:=false', 'record:=false']


def _installed(*packages):
    try:
        from ament_index_python.packages import get_package_share_directory
        for package in packages:
            get_package_share_directory(package)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    shutil.which('ros2') is None
    or not _installed('p4p_bringup', 'p4p_camera', 'p4p_imu', 'p4p_serial_bridge', 'dock_control', 'convchart_qos'),
    reason='needs this workspace built over the inference workspace, and the ros2 CLI')


@pytest.fixture(scope='module')
def rclpy_ctx():
    import rclpy
    saved = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    rclpy.init(domain_id=DOMAIN_ID)
    try:
        yield rclpy
    finally:
        rclpy.try_shutdown()
        if saved is None:
            os.environ.pop('ROS_AUTOMATIC_DISCOVERY_RANGE', None)
        else:
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = saved


@pytest.fixture
def fake_mega(tmp_path):
    port = tmp_path / 'ttyFakeMega'
    proc = subprocess.Popen([sys.executable, str(FAKE_MEGA), '--port', str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 10
    while not port.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert port.exists(), 'fake_mega.py did not create its port'
    yield port
    proc.terminate()
    proc.wait(10)


def launch(tmp_path, *args):
    env = {**os.environ, 'ROS_DOMAIN_ID': str(DOMAIN_ID), 'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST'}
    log = open(tmp_path / 'launch.log', 'w')
    return subprocess.Popen(['ros2', 'launch', 'p4p_bringup', 'pi.launch.py', *BENCH, *args], env=env, cwd=tmp_path,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop(proc):
    """Send SIGINT to the launch process alone and return its exit code.

    Signalling only the launch gives each node one SIGINT, where the whole group would get two.

    Args:
        proc: The ros2 launch process.
    """
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        pytest.fail('the launch did not shut down on SIGINT')


def events(tmp_path):
    """Return the (event, executable) pairs in launch order, from the launch's own log.

    Unlike the ROS graph, the log has no discovery cache still holding an earlier test's node.

    Args:
        tmp_path: Test directory holding launch.log.
    """
    markers = (('process started with pid', 'start'), ('process has finished cleanly', 'done'),
               ('process has died', 'died'))
    out = []
    for line in (tmp_path / 'launch.log').read_text().splitlines():
        if not line.startswith(('[INFO] [', '[ERROR] [')):
            continue
        for marker, event in markers:
            if marker in line:
                out.append((event, line.split('[')[2].split('-')[0]))
    return out


def stop_cleanly(proc, tmp_path):
    assert stop(proc) == 0
    died = [name for event, name in events(tmp_path) if event == 'died']
    assert not died, f'{died} did not shut down cleanly on SIGINT'


def started(tmp_path):
    return {name for event, name in events(tmp_path) if event == 'start'}


def position(tmp_path, event, name):
    return events(tmp_path).index((event, name))


class Probe:
    def __init__(self, rclpy):
        from convchart_interfaces.msg import DriveStatus
        from convchart_qos.qos import IMAGE_SUB_QOS
        from dock_msgs.msg import DockStage
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Imu
        self.rclpy = rclpy
        self.node = rclpy.create_node('pi_launch_probe')
        self.images, self.imus, self.status, self.stages = [], [], [], []
        self.node.create_subscription(CompressedImage, '/image', self.images.append, IMAGE_SUB_QOS)
        self.node.create_subscription(Imu, '/imu/data', self.imus.append, qos_profile_sensor_data)
        self.node.create_subscription(DriveStatus, '/drive/status', self.status.append, qos_profile_sensor_data)
        self.node.create_subscription(DockStage, '/dock/stage', self.stages.append, 10)

    def spin_until(self, predicate, proc, what, timeout_s=TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while not predicate():
            if proc.poll() is not None:
                pytest.fail(f'the launch exited ({proc.returncode}) before {what}')
            if time.monotonic() > deadline:
                pytest.fail(f'timed out waiting for {what}')
            self.rclpy.spin_once(self.node, timeout_sec=0.1)

    def close(self):
        self.node.destroy_node()


def test_sensors_start_the_camera_and_imu_only_with_their_own_frame_ids(rclpy_ctx, tmp_path):
    probe = Probe(rclpy_ctx)
    proc = launch(tmp_path, 'stack:=sensors', 'imu_frame_id:=imu_link')
    try:
        probe.spin_until(lambda: len(probe.images) >= 3 and len(probe.imus) >= 20, proc, 'camera and IMU data')
        assert {m.header.frame_id for m in probe.images} == {'pi_camera'}
        assert {m.header.frame_id for m in probe.imus} == {'imu_link'}
        assert started(tmp_path) == {'camera_node', 'imu_node'}
        stop_cleanly(proc, tmp_path)
    finally:
        stop(proc)
        probe.close()


def test_drive_starts_the_bridge_only_once_the_imu_streams(rclpy_ctx, tmp_path, fake_mega):
    probe = Probe(rclpy_ctx)
    proc = launch(tmp_path, 'stack:=drive', 'camera:=false', f'port:={fake_mega}')
    try:
        probe.spin_until(lambda: any(m.state >= 2 for m in probe.status), proc, 'the bridge reporting the Mega')
        assert started(tmp_path) == {'imu_node', 'wait_for_topic', 'serial_bridge'}
        assert position(tmp_path, 'done', 'wait_for_topic') < position(tmp_path, 'start', 'serial_bridge')
        stop_cleanly(proc, tmp_path)
    finally:
        stop(proc)
        probe.close()


def test_full_on_the_fake_mega_starts_dock_control_once_the_drive_reports_and_can_arm(rclpy_ctx, tmp_path):
    from std_srvs.srv import SetBool
    probe = Probe(rclpy_ctx)
    proc = launch(tmp_path, 'stack:=full', 'camera:=false', 'drive:=fake', f'fake_mega_port:={tmp_path / "tty"}')
    try:
        probe.spin_until(lambda: len(probe.stages) >= 5, proc, 'dock/stage from the MPC')
        assert {'fake_mega', 'ekf_node', 'mpc_node', 'serial_bridge', 'imu_node'} <= started(tmp_path)
        log = events(tmp_path)
        gates = [i for i, e in enumerate(log) if e == ('done', 'wait_for_topic')]
        assert len(gates) == 2, 'the IMU gate, then the drive gate'
        assert gates[0] < log.index(('start', 'serial_bridge')) < gates[1] < log.index(('start', 'ekf_node'))
        assert all(m.command_enabled is False for m in probe.stages), 'nothing may be armed by the launch'
        assert 'nothing is armed' in (tmp_path / 'launch.log').read_text()
        assert all(m.state < 3 for m in probe.status), 'the drive must start disarmed'
        arm = probe.node.create_client(SetBool, '/serial_bridge/arm')
        assert arm.wait_for_service(timeout_sec=10)
        future = arm.call_async(SetBool.Request(data=True))
        probe.spin_until(future.done, proc, 'the arm reply')
        assert future.result().success
        probe.spin_until(lambda: probe.status and probe.status[-1].state >= 3, proc, 'the fake drive armed')
        stop_cleanly(proc, tmp_path)
    finally:
        stop(proc)
        probe.close()


def test_full_without_a_drive_starts_dock_control_after_the_imu_alone(rclpy_ctx, tmp_path):
    probe = Probe(rclpy_ctx)
    proc = launch(tmp_path, 'stack:=full', 'camera:=false', 'drive:=none')
    try:
        probe.spin_until(lambda: len(probe.stages) >= 5, proc, 'dock/stage from the MPC')
        assert {'ekf_node', 'mpc_node', 'imu_node'} <= started(tmp_path)
        assert not {'serial_bridge', 'fake_mega'} & started(tmp_path)
        assert position(tmp_path, 'done', 'wait_for_topic') < position(tmp_path, 'start', 'ekf_node')
        assert all(m.command_enabled is False for m in probe.stages)
        stop_cleanly(proc, tmp_path)
    finally:
        stop(proc)
        probe.close()


def test_a_drive_link_that_never_reports_stops_the_launch_before_dock_control(tmp_path):
    proc = launch(tmp_path, 'stack:=full', 'camera:=false', 'port:=/dev/nonexistent_mega', 'stage_timeout_s:=5.0')
    try:
        proc.wait(TIMEOUT_S)
    finally:
        stop(proc)
    log = (tmp_path / 'launch.log').read_text()
    assert 'the drive link (/drive/status) never came up' in log
    assert 'with state >= 2' in log
    assert not {'ekf_node', 'mpc_node'} & started(tmp_path)
