#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# 输出路径
out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'vi_coordinate_axes.png')

# 定义 IMU 和 Camera 轴
R_i2c = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)

imu_axes = np.eye(3)
cam_axes = R_i2c @ imu_axes

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

origin = np.zeros(3)
scale = 0.6

# IMU axes (实线)
colors = ['r','g','b']
labels_imu = ['IMU X (右)','IMU Y (前)','IMU Z (上)']
for i in range(3):
    v = imu_axes[:,i] * scale
    ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], color=colors[i], arrow_length_ratio=0.1)
    ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, labels_imu[i], color=colors[i])

# Camera axes (虚线)
labels_cam = ['Cam X (右)','Cam Y (下)','Cam Z (前)']
for i in range(3):
    v = cam_axes[:,i] * scale
    ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], color=colors[i], linestyles='dashed', arrow_length_ratio=0.1)
    ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, labels_cam[i], color=colors[i])

# 可视化参考文字
ax.set_title('IMU -> Camera 坐标系映射 (R_i2c = rot +90deg about X)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.8,0.8)
ax.set_ylim(-0.8,0.8)
ax.set_zlim(-0.8,0.8)
ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print('Saved:', out_path)
