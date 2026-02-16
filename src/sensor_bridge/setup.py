from setuptools import find_packages, setup

package_name = 'sensor_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
    'setuptools',
    'pyserial',  # 串口通信库
    'rclpy',     # ROS2 Python 客户端库
    'sensor_msgs', # ROS2 传感器消息包
    ],
    zip_safe=True,
    maintainer='Zhang Lei',
    maintainer_email='32032112@qq.com',
    description='Bridge package for IMU sensor integration and RPY-to-Quaternion conversion.',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'imu_driver = sensor_bridge.imu_driver_node:main',
        ],
    },
)
