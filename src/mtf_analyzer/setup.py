from setuptools import find_packages, setup

package_name = 'mtf_analyzer'
description = (
    'A ROS2 package for MTF (Modulation Transfer Function) analysis, '
    'including target detection using YOLO, UDP-based remote computing, '
    'and lossless image capture for optical testing.'
)
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
        'rclpy',              # ROS2 Python客户端库
        'sensor_msgs',        # ROS2传感器消息类型
        'cv_bridge',          # OpenCV与ROS图像转换工具
        'opencv-python',      # OpenCV库
        'ultralytics',        # YOLO模型推理库
        'numpy',              # 数值计算库
        'scipy',              # 科学计算库（用于插值）
    ],
    zip_safe=True,
    maintainer='Zhang Lei',
    maintainer_email='32032112@qq.com',
    description=description,
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mtf_detector_udp_node = mtf_analyzer.mtf_detector_udp_node:main',
            'mtf_sample_capture = mtf_analyzer.mtf_sample_capture:main',
        ],
    },
)
