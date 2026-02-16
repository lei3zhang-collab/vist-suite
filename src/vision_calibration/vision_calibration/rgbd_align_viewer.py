#!/usr/bin/env python3
"""
Astra Pro Stereo Alignment & Color-to-Depth Registration Verification Node

This ROS2 node provides a real-time visualization tool to verify the extrinsic 
alignment between RGB, Infrared (IR), and Depth streams. It implements two 
registration techniques:
1. Dynamic Re-projection: Uses per-pixel depth data to map IR/Depth to RGB space.
2. Static Homography: A high-speed mapping based on a fixed focal plane (600mm).
The node allows users to interactively adjust transparency and toggle between 
different alignment modes to evaluate calibration accuracy.

功能概述 (Functionality Overview):
-------------------------------
1. 标定参数加载: 从 YAML 文件加载内外参 (K_rgb, D_rgb, K_ir, D_ir, R, T)。
2. 多源同步订阅: 使用 MessageFilter 同步 RGB、Depth 和 IR 三路图像流。
3. 动态对齐 (Dynamic): 利用实时深度图进行逐像素重投影，理论上支持全量程对齐。
4. 固定对齐 (Fixed): 基于 600mm 平面预计算单应性矩阵 (H)，适用于无深度图时的快速验证。
5. olor-to-Depth Registration（将彩色图注册到深度/红外坐标系）
6. 交互式调试:
   - 'M' 键: 切换 [动态深度] 与 [固定600mm] 模式。
   - 'R' 键: 开启/关闭对齐处理（对比原始状态）。
   - 'D' 键: 切换底图（红外图 / 深度伪彩色图）。
   - 'A/Z' 键: 调节 RGB 叠加层的透明度 (Alpha)。

可配置参数 (Inputs):
------------------
- calibration_params.yaml: 必须包含 K_rgb, D_rgb, K_ir, D_ir, R, T 等矩阵。

作者: Zhang Lei
日期: 2026-02-14
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
import os

class AlignedMultiViewer(Node):
    def __init__(self):
        super().__init__('aligned_multi_viewer')
        self.bridge = CvBridge()
        self.skip_display = False # 用于锁定模式下跳过显示，提升性能
        # --- 1. 参数加载 ---
        # 目标路径: ./data/calibration_params/calibration_params.yaml
        # 逻辑：无论在哪里运行，都尝试回溯到 workspace 根目录寻找 data 文件夹
        self.calib_params = self.get_robust_config_path('calibration_params/calibration_params.yaml')
        self.prepare_matrices()
        self.width, self.height = 640, 480

        # --- 2. 预计算固定深度 600mm 的 H 矩阵 ---
        self.Z_fixed = 600.0
        self.update_fixed_h_matrix()

        # 状态开关
        self.remap_enabled = True 
        self.alignment_mode = "DYNAMIC" # "DYNAMIC" 或 "FIXED_600"
        self.mode_use_depth = False 
        self.alpha = 0.5
        
        self.win_name = "Stereo_Alignment_Verification"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 1280, 960)

        # --- 3. ROS 2 订阅 ---
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ir_sub = message_filters.Subscriber(self, Image, '/camera/ir/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.ir_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info(">>> 系统启动 <<<")
        self.get_logger().info("按 'M' 切换模式: [动态深度] <-> [固定600mm]")
        self.get_logger().info("按 'R' 开启/关闭对齐 | 'D' 切换底图 | 'A/Z' 透明度")
    def get_robust_config_path(self, relative_path):
        """
        技术关键：项目锚点定位法
        1. 获取当前脚本绝对路径
        2. 向上寻找 'src' 文件夹作为锚点
        3. 定位到 src 同级的 'data' 目录
        """
        # A. 获取脚本所在绝对路径 (例如: /home/user/ros2_ws/src/my_pkg/my_pkg/script.py)
        script_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(script_path)
        
        # B. 向上追溯，直到找到 'src' 或到达根目录
        # 这是为了定位到 Workspace 根目录
        ws_root = current_dir
        while ws_root != '/' and not os.path.exists(os.path.join(ws_root, 'src')):
            ws_root = os.path.dirname(ws_root)
        
        # C. 构造开发环境下的目标路径: workspace_root/data/calibration_params/...
        path_dev = os.path.join(ws_root, 'data', relative_path)
        
        # D. 备选方案：ROS2 安装目录 (share/package/data/...)
        path_ros = ""
        try:
            # 获取包名（假设与文件夹同名或通过 node 名获取）
            pkg_name = self.get_name() 
            path_ros = os.path.join(get_package_share_directory(pkg_name), 'data', relative_path)
        except Exception:
            pass

        # 优先级判断与加载
        final_path = ""
        if os.path.exists(path_dev):
            final_path = path_dev
            self.get_logger().info(f">>> 开发模式：定位到 Workspace 数据: {final_path}")
        elif path_ros and os.path.exists(path_ros):
            final_path = path_ros
            self.get_logger().info(f">>> 部署模式：定位到安装目录数据: {final_path}")
        else:
            # 最后的尝试：直接拼接当前目录
            final_path = os.path.join('.', 'data', relative_path)
            self.get_logger().warn(f">>> 警告：未找到项目锚点，尝试相对路径: {os.path.abspath(final_path)}")

        if not os.path.exists(final_path):
            self.get_logger().error(f"找不到标定文件: {final_path}")
            return {}

        try:
            with open(final_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f"YAML 解析失败: {e}")
            return {}
    def load_calibration_params(self, filename):
        if not os.path.exists(filename):
            self.get_logger().error(f"未找到 {filename}")
            return {}
        with open(filename, 'r') as f:
            return yaml.safe_load(f)

    def prepare_matrices(self):
        self.K_rgb = np.array(self.calib_params['K_rgb'], dtype=np.float64)
        self.D_rgb = np.array(self.calib_params['D_rgb'], dtype=np.float64)
        self.K_ir = np.array(self.calib_params['K_ir'], dtype=np.float64)
        self.D_ir = np.array(self.calib_params['D_ir'], dtype=np.float64)
        self.R = np.array(self.calib_params['R'], dtype=np.float64)
        self.T = np.array(self.calib_params['T'], dtype=np.float64).reshape(3, 1)

    def update_fixed_h_matrix(self):
        # 计算固定 600mm 的单向投影矩阵 H
     
        K1_inv = np.linalg.inv(self.K_ir)
        R_comb = self.R + (self.T / self.Z_fixed) @ np.array([[0, 0, 1]])
        self.H_fixed = self.K_rgb @ R_comb @ K1_inv
        
        # 预生成静态映射表提速
        grid_y, grid_x = np.mgrid[0:480, 0:640].astype(np.float32)
        pix = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3).T
        new_pix = self.H_fixed @ pix
        new_pix /= new_pix[2, :]
        self.static_map_x = new_pix[0, :].reshape(480, 640).astype(np.float32)
        self.static_map_y = new_pix[1, :].reshape(480, 640).astype(np.float32)

    def sync_callback(self, color_msg, depth_msg, ir_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        ir_raw = self.bridge.imgmsg_to_cv2(ir_msg, "mono8")

        self.latest_depth = depth_raw # 保存深度图像为实例属性，供子类使用

        # 1. 准备底图
        if self.mode_use_depth:
            depth_vis = cv2.normalize(np.clip(depth_raw, 500, 4000), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            source_img = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        else:
            source_img = cv2.cvtColor(cv2.equalizeHist(ir_raw), cv2.COLOR_GRAY2BGR)

        # 2. 对齐逻辑
        if not self.remap_enabled:
            overlay = cv2.addWeighted(source_img, 1-self.alpha, color_img, self.alpha, 0)
            mode_str = "OFF (RAW)"
        else:
            if self.alignment_mode == "FIXED_600":
                # 使用预计算的 600mm 映射表 进行快速重映射
                aligned_color = cv2.remap(color_img, self.static_map_x, self.static_map_y, cv2.INTER_LINEAR)
                mode_str = "FIXED 600mm"
            else:
                # 动态深度逐像素重投影
                h, w = depth_raw.shape
                v, u = np.mgrid[0:h, 0:w]
                valid = depth_raw > 0
                z = depth_raw[valid].astype(np.float32) #布尔索引，降维到1维
                # 计算 IR 坐标系下的 3D 点
                x_ir = (u[valid] - self.K_ir[0, 2]) * z / self.K_ir[0, 0]
                y_ir = (v[valid] - self.K_ir[1, 2]) * z / self.K_ir[1, 1]
                p_ir = np.stack([x_ir, y_ir, z], axis=-1)   #p_ir 是 N x 3 的点云数据，N 是有效深度像素的数量
                # 将 IR 坐标系下的点转换到 RGB 坐标系下

                p_rgb = p_ir @ self.R.T + self.T.T
                # 计算 RGB 图像平面上的像素坐标
                z_rgb = p_rgb[:, 2]
                z_rgb[z_rgb == 0] = 1.0 # 避免除零错误
                u_rgb = (p_rgb[:, 0] * self.K_rgb[0, 0] / z_rgb + self.K_rgb[0, 2]).astype(np.int32)
                v_rgb = (p_rgb[:, 1] * self.K_rgb[1, 1] / z_rgb + self.K_rgb[1, 2]).astype(np.int32)
                # 过滤掉投影到 RGB 图像外的点
                valid_rgb = (u_rgb >= 0) & (u_rgb < w) & (v_rgb >= 0) & (v_rgb < h)
                aligned_color = np.zeros_like(color_img)
                # 将 RGB 图像的颜色值映射到 IR 图像坐标系下
                aligned_color[v[valid][valid_rgb], u[valid][valid_rgb]] = color_img[v_rgb[valid_rgb], u_rgb[valid_rgb]]
                mode_str = "DYNAMIC DEPTH"

            # 统一去畸变基础图, 底图是 IR 图，叠加是 RGB 图
            ir_base = cv2.undistort(source_img, self.K_ir, self.D_ir)
            overlay = cv2.addWeighted(ir_base, 1-self.alpha, aligned_color, self.alpha, 0)

        # 3. 信息显示
        cv2.putText(overlay, f"Mode: {mode_str} | Alpha: {self.alpha:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.overlay = overlay # 将 overlay 保存为实例属性，以便在锁定模式下使用
        if hasattr(self, 'skip_display') and self.skip_display:
            pass
        else:
            cv2.imshow(self.win_name, overlay)
        
        # 4. 键盘响应
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): raise KeyboardInterrupt
        elif key == ord('r'): self.remap_enabled = not self.remap_enabled
        elif key == ord('m'): # 切换模式
            self.alignment_mode = "FIXED_600" if self.alignment_mode == "DYNAMIC" else "DYNAMIC"
            self.get_logger().info(f"切换至: {self.alignment_mode}")
        elif key == ord('d'): self.mode_use_depth = not self.mode_use_depth
        elif key == ord('a'): self.alpha = min(self.alpha + 0.1, 1.0)
        elif key == ord('z'): self.alpha = max(self.alpha - 0.1, 0.0)
def main():
    rclpy.init()
    node = AlignedMultiViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("正在退出...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()