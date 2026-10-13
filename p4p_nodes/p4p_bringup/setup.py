from setuptools import setup

package_name = 'p4p_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pi.launch.py']),
        ('share/' + package_name + '/tools', ['../../tools/fake_mega.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='dev@todo.todo',
    description="The Pi's ordered bring-up: camera and IMU, then the Arduino bridge, then dock_control",
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'wait_for_topic = p4p_bringup.wait_for_topic:main',
        ],
    },
)
