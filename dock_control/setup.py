from glob import glob

from setuptools import find_packages, setup

package_name = 'dock_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='kgra470@aucklanduni.ac.nz',
    description='Docking estimator (EKF), supervisor and MPC for the holonomic demonstrator.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # One process each, pinned to its own core (brief Section 3).
            'ekf_node = dock_control.ekf_node:main',
            'mpc_node = dock_control.mpc_node:main',
        ],
    },
)
