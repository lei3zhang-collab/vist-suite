#!/usr/bin/env python3
"""
MTF Detection and Remote Computing Node (YOLO + UDP)

This node utilizes a YOLO model to detect 9-point MTF targets in real-time. 
It extracts raw luminance data from regions of interest (ROIs), transmits 
the data to a remote PC via UDP for high-precision computation, and 
visualizes the returned MTF values as a dynamic heatmap.

功能概述 (Functionality Overview):
-------------------------------
1. 视觉识别：加载 ONNX 格式的 YOLO 模型，识别视野中的 9 个标靶。
2. 逻辑过滤：通过中心 NMS (Non-Maximum Suppression) 和空间排序确保 9 点定位准确。
3. 原始数据传输：提取 150x150 的灰度 ROI 区域，并通过 UDP 协议分包发送至计算端。
4. 闭环反馈：接收 PC 端计算出的 MTF 数值。
5. 热力图可视化：利用 Scipy 的 Griddata 线性插值生成全视野 MTF 分布云图。
注意事项：
- 请确保 Astra Pro 摄像头已正确连接
- 通过下面的命令调整摄像头参数以优化图像质量：
v4l2-ctl -d /dev/video0 -c sharpness=1 -c backlight_compensation=0 -c white_balance_automatic=0
- 分辨率更改为1280x720，并启动摄像头
ros2 launch astra_camera astra_pro.launch.xml color_width:=1280 color_height:=720

作者: Zhang Lei
日期: 2026-02-15
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import socket
from scipy.interpolate import griddata


class MtfYoloViewer(Node):
    def __init__(self):
        super().__init__('mtf_yolo_viewer')
        
        # 1. YOLO 模型配置
        # 指定 ONNX 模型路径，适用于在边缘设备上推理
        self.model_path = '/home/yahboom/workspace/src/test/best_720.onnx' 
        self.model = YOLO(self.model_path, task='detect')
        self.dist_thresh = 50.0 
        
        # 2. UDP 通讯配置 (用于与上位机分析软件交互)
        self.pc_ip = "192.168.0.115"  # 远程计算 PC 的 IP 地址
        self.pc_port = 5005           # 发送端口
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind(("0.0.0.0", 5006))
        self.recv_sock.settimeout(1)    # 设置 1 秒超时，防止网络波动导致主线程阻塞

        # 增加发送缓冲区，防止高分辨率图像数据包丢失
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        
        # 3. 图像处理参数
        self.crop_size = 150    # 标靶切片尺寸 (像素)
        self.send_trigger = False
        
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.listener_callback, 10)
        
        self.get_logger().info('MTF 亮度原始数据发送版已启动。按 "s" 发送数据。')

    def sort_boxes_with_scores(self, boxes, scores):
        """ 将检测到的 9 个框按照 3x3 矩阵顺序重排 (从左到右，从上到下) """
        if len(boxes) != 9: return boxes, scores
        # 计算中心点
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        # 首先按 Y 轴排序 (分出三行)
        y_idx = np.argsort(centers[:, 1])
        boxes_y, scores_y, centers_y = boxes[y_idx], scores[y_idx], centers[y_idx]
        
        final_boxes, final_scores = [], []
        # 对每一行内部按 X 轴排序
        for i in range(0, 9, 3):
            x_idx = np.argsort(centers_y[i:i+3, 0])
            final_boxes.append(boxes_y[i:i+3][x_idx])
            final_scores.append(scores_y[i:i+3][x_idx])
        return np.vstack(final_boxes), np.concatenate(final_scores)

    def apply_center_nms(self, results, dist_thresh):
        """ 基于中心点距离的非极大值抑制，防止同一标靶出现重叠框 """
        if len(results[0].boxes) == 0: return np.array([]), np.array([])
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
        order = scores.argsort()[::-1]
        keep, skipped = [], set()
        for i in range(len(order)):
            idx = order[i]
            if idx in skipped: continue
            keep.append(idx)
            for j in range(i + 1, len(order)):
                if np.linalg.norm(centers[idx] - centers[order[j]]) < dist_thresh:
                    skipped.add(order[j])
        f_boxes, f_scores = boxes[keep], scores[keep]
        # 如果找齐了 9 个框，进行空间排序
        if len(f_boxes) == 9: f_boxes, f_scores = self.sort_boxes_with_scores(f_boxes, f_scores)
        return f_boxes, f_scores

    def process_and_send_raw(self, cv_image, boxes):
        """
        提取 9 个 150x150 区域的亮度信息并分包发送原始字节
        """
        h, w = cv_image.shape[:2]
        half = self.crop_size // 2
        
        # 1. 虚拟机内先算好亮度 (灰度化)
        gray_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 2. 拼接 9 个 ROI 的原始字节
        raw_payload = bytearray()
        for box in boxes:
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            x1, y1 = max(0, cx - half), max(0, cy - half)
            # 确保切片大小固定为 150x150
            roi = gray_img[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size]
            
            # 如果靠边导致尺寸不足，填充黑色
            if roi.shape[0] != self.crop_size or roi.shape[1] != self.crop_size:
                roi = cv2.copyMakeBorder(roi, 0, self.crop_size - roi.shape[0], 
                                         0, self.crop_size - roi.shape[1], 
                                         cv2.BORDER_CONSTANT, value=0)
            raw_payload.extend(roi.tobytes())

        # 3. 分包发送 (每包约 50KB)
        total_len = len(raw_payload)
        chunk_size = 50000 
        for offset in range(0, total_len, chunk_size):
            chunk = raw_payload[offset : offset + chunk_size]
            # 包头: 标志 0xEE (1字节) + 偏移 (4字节) + 总长度 (4字节)
            header = b'\xEE' + offset.to_bytes(4, 'big') + total_len.to_bytes(4, 'big')
            self.udp_sock.sendto(header + chunk, (self.pc_ip, self.pc_port))
            
        self.get_logger().info(f'已完成亮度提取并分包发送，总字节: {total_len}')


    def draw_mtf_heatmap(self, display_img, boxes, mtf_values, contrast_factor=2.5):
        h, w = display_img.shape[:2]
        """
        在输入图像上绘制MTF（调制传递函数）热力图，并叠加原始图像。

        参数:
            display_img (numpy.ndarray): 输入的背景图像，形状为 (H, W, 3)。
            boxes (list or numpy.ndarray): 检测框列表，每个框为 [x1, y1, x2, y2] 格式。
            mtf_values (list or numpy.ndarray): 每个检测框对应的MTF值。
            contrast_factor (float, optional): 对比度增强因子，默认为2.5。

        返回:
            numpy.ndarray: 叠加热力图后的图像，形状与display_img相同。
        """
        # 1. 获取 9 个点的中心位置
        points = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            points.append([(x1 + x2) / 2, (y1 + y2) / 2])
        points = np.array(points, dtype=np.float32)

        # 2. 准备插值数据
        ref_mtf = 0.62
        norm_values = (np.array(mtf_values) - ref_mtf) * contrast_factor + 0.5
        norm_values = np.clip(norm_values * 255, 0, 255).astype(np.float32)

        # 3. 全图网格插值
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        # 使用 linear 插值
        heatmap_gray = griddata(points, norm_values, (grid_x, grid_y), method='linear', fill_value=0)
        heatmap_gray = np.nan_to_num(heatmap_gray).astype(np.uint8)

        # 4. 【核心新增】：创建区域遮罩 (Mask)
        # 找到 9 个点的凸包（外围轮廓）
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.drawContours(mask, [hull], -1, 255, -1) # 在 Mask 上画一个实心的白色多边形

        # 5. 伪彩色映射
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        # 平滑处理
        heatmap_color = cv2.GaussianBlur(heatmap_color, (31, 31), 0)

        # 6. 只保留 Mask 区域内的热力图
        # 没采样的地方保持原图，有采样的地方做叠加
        overlay = display_img.copy()
        alpha = 0.5
        
        # 仅对 Mask 区域进行加权融合
        roi_indices = np.where(mask == 255)
        overlay[roi_indices] = cv2.addWeighted(
            display_img[roi_indices], 1.0 - alpha, 
            heatmap_color[roi_indices], alpha, 0
        )

        # 7. 增强显示：画出 9 个 Box 的边框和数值
        for i, (box, mtf) in enumerate(zip(boxes, mtf_values)):
            x1, y1, x2, y2 = box.astype(int)
            # 画个小框显眼点
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(overlay, f"MTF:{mtf:.3f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return overlay
    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 运行 YOLO 推理
            results = self.model.predict(source=cv_image, device='cpu', imgsz=[736, 1280], conf=0.05, verbose=False)
            # 应用中心点非极大值抑制，筛选最终的边界框和得分
            final_boxes, final_scores = self.apply_center_nms(results, self.dist_thresh)  
            
            display_img = cv_image.copy()
            # 绘制实时检测框
            for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"#{i+1}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 判断是否需要触发UDP通信
            if self.send_trigger:
                if len(final_boxes) == 9:
                    self.process_and_send_raw(cv_image, final_boxes)
                    try:
                        # 等待并解析 PC 端返回的 MTF 数组 (Float32)
                        data, _ = self.recv_sock.recvfrom(1024)
                        mtf_res = np.frombuffer(data, dtype=np.float32)
                        
                        # 成功接收后生成并叠加分析热力图
                        display_img = self.draw_mtf_heatmap(display_img, final_boxes, mtf_res)
                        self.get_logger().info(f'MTF计算成功: {mtf_res.mean():.4f}')
                        
                    except socket.timeout:
                        # 捕获超时异常，不让程序挂掉
                        self.get_logger().error('UDP 接收超时：PC端未响应')
                    except Exception as e:
                        # 捕获其他网络异常（如连接重置等）
                        self.get_logger().error(f'UDP 通讯错误: {e}')
                else:
                    self.get_logger().warn('9图未找全，取消发送')
                
                # 无论成功失败，都重置触发位，防止死循环发送
                self.send_trigger = False

            cv2.imshow("MTF_Viewer", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): self.send_trigger = True
            elif key == ord('q'):
                self.get_logger().info("用户请求退出")
                raise KeyboardInterrupt
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main():
    rclpy.init()
    node = MtfYoloViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.udp_sock.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()