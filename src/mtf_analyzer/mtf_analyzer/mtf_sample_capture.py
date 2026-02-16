#！/usr/bin/env python3
"""
Astra Pro Image Capture and Lossless Storage Node (ImageSaver)

This node is designed for MTF (Modulation Transfer Function) sample collection.
It subscribes to ROS2 image topics, providing real-time preview, interactive 
lossless snapshots, and automated directory management for optical testing.

本节点专为 MTF（调制传递函数）样本采集而设计。
它订阅 ROS2 图像话题，提供实时预览、交互式无损快照以及用于光学测试的自动化目录管理。

技术概述：
- 话题：订阅 /camera/color/image_raw (BGR8)。
- 存储：以零压缩的 PNG 格式保存，以确保数据完整性。
- 交互：按 'C' 键捕获，按 'Q' 键安全退出。
注意事项：
- 请确保 Astra Pro 摄像头已正确连接
- 通过下面的命令调整摄像头参数以优化图像质量：
v4l2-ctl -d /dev/video0 -c sharpness=1 -c backlight_compensation=0 -c white_balance_automatic=0
- 分辨率更改为1280x720，并启动摄像头
ros2 launch astra_camera astra_pro.launch.xml color_width:=1280 color_height:=720

作者: Zhang Lei
日期: 2026-02-14
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        # 1. 订阅话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        
        self.bridge = CvBridge()
        self.latest_image = None
        
        # 2. 创建文件夹
        home = str(Path.home())
        default_path = os.path.join(home, 'workspace_Ubuntu', 'data', 'mtf_samples')
        self.declare_parameter('mtf_save_path', default_path)
        self.save_path = self.get_parameter('mtf_save_path').get_parameter_value().string_value
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.get_logger().info(f"保存图片路径: {self.save_path}")
        self.get_logger().info("程序已启动。请在弹出的窗口中按下 'c' 键截屏，按下 'q' 键退出。")

    def listener_callback(self, msg):
        # 将 ROS 图像转换为 OpenCV 格式
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # 实时显示预览图
        cv2.imshow("Camera Preview", self.latest_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.save_image()
        elif key == ord('q'):
            self.get_logger().info("用户请求退出")
            raise KeyboardInterrupt

    def save_image(self):
        if self.latest_image is not None:
            # 3. 自动递增文件名
            existing_files = [f for f in os.listdir(self.save_path) if f.endswith('.png')]
            count = len(existing_files) + 1
            file_name = f"mtf_sample_{count:03d}.png"
            full_path = os.path.join(self.save_path, file_name)
            
            # 4. 保存为无损 PNG (compression级别设为0表示无损压缩最快)
            cv2.imwrite(full_path, self.latest_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.get_logger().info(f"已保存无损图片: {full_path}")
        else:
            self.get_logger().warn("尚未接收到图像数据！")

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImageSaver()
    try:
        rclpy.spin(image_saver)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        image_saver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()