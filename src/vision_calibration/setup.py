from setuptools import find_packages, setup

package_name = 'vision_calibration'

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
    'opencv-python>=4.5.0',
    'numpy>=1.20.0',
    'pyyaml',
    ],
    zip_safe=True,
    maintainer='Zhang Lei',
    maintainer_email='32032112@qq.com',
    description='A ROS2 package for stereo camera calibration and RGB-D alignment, including tools for data collection, intrinsic/extrinsic parameter computation, and real-time visualization and vi target lock.',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_collector = vision_calibration.stereo_data_collector:main',
            'stereo_calibrator = vision_calibration.scripts.stereo_calibrator_script:main',
            'stereo_visualizer = vision_calibration.rgbd_align_viewer:main',
            'vi_target_lock = vision_calibration.vi_target_lock_node:main',
        ],
    },
)
