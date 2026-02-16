#！/usr/bin/env python3
"""
Stereo Camera Calibration Data Collector

This ROS2 node collects synchronized RGB and IR images from a stereo camera system
for camera calibration purposes. It detects chessboard corners in both image streams,
displays the detection results, and saves the image pairs when triggered by the user.

功能概述 (Functionality Overview):
-------------------------------
0. 建议打印棋盘格标靶，关闭或者遮蔽红外结构光发生器，并采用红外补光灯补光，提升角点检测成功率。
1. 订阅RGB和IR图像话题，使用时间同步器确保图像对齐。
2. 实时显示图像并检测棋盘格角点。
3. 操作说明：
   - 按 'S' 键：自动检测角点并保存图像对。
   - 按 'F' 键：强制保存当前帧。
   - 按 'Q' 键：退出程序。
4. 图像保存路径可配置，默认为 './data/calib_raw'。
5. 支持实时预览和交互操作。

可配置参数 (Configurable Parameters):
-------------------------------------
1. board_size (tuple, default: (9, 6))
   - 棋盘格内角点数量（列数, 行数）。
2. save_path (string, default: './data/calib_raw')
   - 图像保存路径。

作者: Zhang Lei
日期: 2026-02-14
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
import os
import numpy as np
from pathlib import Path

class CalibrationDataCollector(Node):
    def __init__(self):
        super().__init__('calib_collector')
        self.bridge = CvBridge()
        
        # 棋盘格规格 (内角点数量)
        self.board_size = (9, 6)
        # self.save_path = './calib_data' # 保存路径
        home = str(Path.home())
        default_path = os.path.join(home, 'workspace_Ubuntu', 'data', 'calib_raw')
        self.declare_parameter('save_path', default_path)
        self.save_path = self.get_parameter('save_path').get_parameter_value().string_value

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # 订阅话题 (请根据 ros2 topic list 实际名称修改)
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.ir_sub = message_filters.Subscriber(self, Image, '/camera/ir/image_raw')
        
        # 同步两路图像流
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.ir_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        self.img_id = 0
        self.get_logger().info(">>> 标定采集系统启动 <<<")
        self.get_logger().info("按 'S' 键：自动检测并保存 (更精准)")
        self.get_logger().info("按 'F' 键：强制保存当前帧 (用于无补光灯环境)")

    def sync_callback(self, color_msg, ir_msg):
        # 1. 转换图像
        color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # 处理 IR 图：Astra Pro 的 IR 是 16位，先转成 8位进行显示和处理
        ir_raw = self.bridge.imgmsg_to_cv2(ir_msg, "mono8")
        
        # 2. 图像增强 (仅用于预览显示，不影响保存的原始数据)
        # 如果没灯很暗，做一次直方图均衡化让肉眼能看清
        ir_enhanced = cv2.equalizeHist(ir_raw)

        # 3. 尝试角点检测
        ret_c, corners_c = cv2.findChessboardCorners(color_img, self.board_size)
        ret_i, corners_i = cv2.findChessboardCorners(ir_enhanced, self.board_size)

        # 4. 绘制预览窗
        vis_color = color_img.copy()
        if ret_c:
            cv2.drawChessboardCorners(vis_color, self.board_size, corners_c, ret_c)
        
        vis_ir = cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2BGR)
        if ret_i:
            cv2.drawChessboardCorners(vis_ir, self.board_size, corners_i, ret_i)

        # 将两张图拼在一起显示
        h, w = color_img.shape[:2]
        ir_resized = cv2.resize(vis_ir, (w, h))
        display_img = np.hstack((vis_color, ir_resized))
        
        cv2.putText(display_img, f"Captured: {self.img_id}", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calibration - [S] Smart Save | [F] Force Save", display_img)
        
        key = cv2.waitKey(1) & 0xFF

        # 5. 保存逻辑
        if key == ord('s'): # 智能保存
            if ret_c and ret_i:
                self.save_images(color_img, ir_raw, "smart")
            else:
                self.get_logger().warn("检测失败！无法自动识别角点，请调整角度或按 'F' 强制采集")
        
        elif key == ord('f'): # 强制保存
            self.save_images(color_img, ir_raw, "force")
        elif key == ord('q'):
            self.get_logger().info("用户请求退出")
            raise KeyboardInterrupt

    def save_images(self, color, ir, mode):
        self.img_id += 1
        c_name = f"{self.save_path}/rgb_{self.img_id:02d}.png"
        i_name = f"{self.save_path}/ir_{self.img_id:02d}.png"
        cv2.imwrite(c_name, color)
        cv2.imwrite(i_name, ir)
        self.get_logger().info(f"[{mode.upper()}] 已保存第 {self.img_id} 组图像对")

def main():
    rclpy.init()
    node = CalibrationDataCollector()
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