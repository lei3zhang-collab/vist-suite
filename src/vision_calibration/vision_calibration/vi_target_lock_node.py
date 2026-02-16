#!/usr/bin/env python3
# vi_target_lock_node.py
"""
Visual-Inertial Target Locking Node for Astra Pro Stereo Camera

The `VITargetLockNode` is a high-level perception component inheriting from `AlignedMultiViewer`. 
It extends basic RGB-D alignment with Visual-Inertial (VI) fusion to maintain target 
persistence during rapid camera ego-motion. 

功能概述 (Functionality Overview):
-------------------------------
1. 视觉-惯性融合锁定: 结合 Astra Pro 的深度感知能力与 IMU 的姿态变化，实现 3D 空间点的实时跟踪。
2. 坐标系对齐: 解决了 IMU 机体系 (Body Frame) 与相机光学坐标系 (Optical Frame) 之间的坐标轴转换问题。
3. 相对增量预测: 即使相机发生剧烈晃动，也能根据姿态变化量实时重投影锁定点。
4. 交互式锁定: 用户通过鼠标点击选择目标点，系统自动计算并显示预测位置。

技术关键点 (Key Technical Insights):
-------------------------------
- 坐标系定义: 
    IMU (Body): x-右, y-前, z-上
    Camera (Optical): x-右, y-下, z-前
    映射关系: Camera_X=IMU_X, Camera_Y=-IMU_Z, Camera_Z=IMU_Y (绕X轴转90度)
- 相对运动逻辑: 
    由于空间点在世界坐标系中静止，当相机正向旋转时，空间点相对于相机的坐标应进行反向旋转（即转置矩阵运算）。

作者: Zhang Lei
日期: 2026-02-14
"""

from vision_calibration.rgbd_align_viewer import AlignedMultiViewer
from sensor_msgs.msg import Imu
from rclpy.time import Time
import numpy as np
import cv2
import rclpy
from collections import deque
import bisect

class VITargetLockNode(AlignedMultiViewer):
    def __init__(self):
        # 调用父类初始化，自动完成相机内参加载、多话题订阅与同步设置
        super().__init__()
        self.skip_display = True # 用于锁定模式下跳过父类的显示
        # --- 时间同步扩展 ---        # --- 时间同步缓冲区 ---
        # 存储元组 (nanoseconds, R_matrix)，保留最近 2秒的数据 (50Hz * 2 = 100)
        self.imu_buffer = deque(maxlen=100)

        # --- 扩展状态变量 ---
        self.is_locked = False          # 锁定状态标志
        self.R_start_raw = np.eye(3)    # 锁定瞬间的位姿初始值
        self.p_cam_start = None         # 锁定点在 IR 坐标系下的 3D 矢量 [x, y, z]^T

        # --- 坐标系变换常量 ---
        # 核心逻辑：将 IMU 的旋转增量转换到相机视角下，imu 坐标系定义：x轴向右，y轴向前，z轴向上，而相机光学坐标系定义：x轴向右，y轴向下，z轴向前
        # 即绕X轴的旋转90°矩阵为：
        self.R_imu_to_cam = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1, 0]
        ], dtype=np.float64)

        # --- 话题扩展 ---
        # 订阅 IMU 原始数据，用于姿态解算
        self.imu_sub = self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 10)
        
        # 重新绑定鼠标回调
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        self.get_logger().info(">>> VI 目标锁定节点启动 (带时间同步) <<<")

    def quaternion_to_matrix(self, q):
        """ 辅助函数：四元数转旋转矩阵 """
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])

    def imu_callback(self, msg):
        """
        IMU 数据回调：将数据打上时间戳存入缓冲区。
        """
        # 获取消息头的时间戳（纳秒）
        t_ns = Time.from_msg(msg.header.stamp).nanoseconds
        R = self.quaternion_to_matrix(msg.orientation)
        
        # 存入队列 (时间单调递增)
        self.imu_buffer.append((t_ns, R))

    def get_interpolated_imu_pose(self, target_time_ns):
        """
        在缓冲区中通过二分查找寻找最接近图像时间戳的姿态 
        Nearest Neighbor 匹配在 50Hz IMU 频率下能满足大多数手持场景需求
        """
        if not self.imu_buffer:
            return np.eye(3)

        # 提取缓冲区中的所有时间戳
        times = [d[0] for d in self.imu_buffer]
        
        # 二分查找插入点
        idx = bisect.bisect_left(times, target_time_ns)
        
        # 边界处理
        if idx == 0:
            best_idx = 0
        elif idx == len(times):
            best_idx = -1
        else:
            # 比较前后两个点，谁离得近选谁 (Nearest Neighbor)
            dt_before = abs(target_time_ns - times[idx-1])
            dt_after = abs(target_time_ns - times[idx])
            best_idx = idx - 1 if dt_before < dt_after else idx
            
        return self.imu_buffer[best_idx][1]

    def mouse_callback(self, event, x, y, flags, param):
        """ 鼠标点击锁定：捕获当前像素对应的 3D 空间坐标及位姿快照 """
        if event == cv2.EVENT_LBUTTONDOWN and self.latest_depth is not None:
            z_mm = self.latest_depth[y, x]
            if z_mm <= 0: return
            z_m = z_mm / 1000.0

            # 1. 图像系 -> 相机 3D 空间系 (反投影)
            # p = Z * K_inv * [u, v, 1]^T
            self.p_cam_start = np.array([
                (x - self.K_ir[0, 2]) * z_mm / self.K_ir[0, 0],
                (y - self.K_ir[1, 2]) * z_mm / self.K_ir[1, 1],
                z_mm
            ]).reshape(3, 1)

            # 2. 捕获锁定瞬间的基准姿态
            if self.imu_buffer:
                self.R_start_raw = self.imu_buffer[-1][1].copy()
            else:
                self.R_start_raw = np.eye(3)
                
            self.is_locked = True
            self.get_logger().info(f"Locked 3D: {self.p_cam_start.flatten()}, Depth: {z_mm}mm")

    def sync_callback(self, color_msg, depth_msg, ir_msg):
        """
        重写父类的同步回调，添加时间对齐逻辑。
        """
        # 1. 先调用父类，完成图像对齐、去畸变和背景渲染
        super().sync_callback(color_msg, depth_msg, ir_msg)

        # 2. 获取当前图像的时间戳
        img_time_ns = Time.from_msg(color_msg.header.stamp).nanoseconds

        # 3. 核心追踪逻辑
        if self.is_locked and self.p_cam_start is not None:
            try:
                # 1. 计算 IMU 机体系下的相对增量旋转
                # R_delta_imu 表示从 start 到 current 在 IMU 自己看来是怎么转的
                if self.imu_buffer:
                    self.R_current_raw = self.get_interpolated_imu_pose(img_time_ns)
                else:
                    self.R_current_raw = np.eye(3)
                R_delta_imu = self.R_start_raw.T @ self.R_current_raw

                # 2. 相似变换：把 IMU 的转动映射到相机的坐标轴上
                # 相似变换公式: R_cam = R_i2c * R_imu * R_i2c^T
                R_delta_cam = self.R_imu_to_cam @ R_delta_imu @ self.R_imu_to_cam.T

                # 3. 预测目标点在当前相机系下的新位置
                # 因為是相機在動，空間點相對相機做「反向旋轉」
                # 所以使用 R_delta_cam 的轉置 (即逆矩陣)
                p_cam_now = R_delta_cam.T @ self.p_cam_start
                tx, ty, tz = p_cam_now.flatten()

                # 4. 重投影：3D 空间 -> 2D 像素平面
                if tz > 0.1: # 避开分母为0
                    u = int(tx * self.K_ir[0, 0] / tz + self.K_ir[0, 2])
                    v = int(ty * self.K_ir[1, 1] / tz + self.K_ir[1, 2])
                    # 在 overlay 层绘制锁定标记
                    if 0 <= u < self.width and 0 <= v < self.height:
                        cv2.drawMarker(self.overlay, (u, v), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
            except Exception as e:
                self.get_logger().error(f"Projection error: {e}")

        # 显示最终融合后的增强现实视图
        cv2.imshow(self.win_name, self.overlay)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = VITargetLockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()