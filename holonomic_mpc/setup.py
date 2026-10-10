from setuptools import find_packages, setup

package_name = 'holonomic_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dev',
    maintainer_email='kgra470@aucklanduni.ac.nz',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'holonomic_mpc = holonomic_mpc.holonomic_mpc:main',
            'ekf = holonomic_mpc.ekf:main',
        ],
    },
)
