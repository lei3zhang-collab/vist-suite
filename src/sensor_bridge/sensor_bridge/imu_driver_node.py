#!/usr/bin/env python3
"""
IMU Driver Node for WIT Motion Sensors (BWT901BLECL5.0 https://ucngu0zfs13q.feishu.cn/wiki/WJXowIO7si0U28kiaTMc17CYnrg)

This ROS2 node interfaces with WIT motion sensors via serial communication,
parses the sensor data according to the WIT standard protocol, and publishes
the data as standardized ROS2 IMU messages.

可配置参数 (Configurable Parameters):
-------------------------------------
1. port (string, default: '/dev/ttyUSB0')
   - 串口设备路径，用于连接IMU传感器。
   - Serial port device path for connecting to the IMU sensor.

功能概述 (Functionality Overview):
-------------------------------
1. 初始化串口连接，支持错误捕获和保护机制。
2. 定时读取串口数据，解析WIT标准协议（包头0x55 0x61，包长20字节）。
3. 提取加速度、角速度和欧拉角数据，并转换为ROS标准单位：
   - 加速度：g → m/s²
   - 角速度：deg/s → rad/s
   - 欧拉角：deg → rad
4. 将数据封装为ROS2标准的Imu消息，发布到/imu/data_raw话题。
5. 支持四元数转换（RPY → Quaternion），符合ROS坐标系标准。
6. 节点销毁时自动关闭串口连接，防止资源占用。

作者: Zhang Lei
日期: 2026-02-14
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial
import struct
import math

class WitBleStandardNode(Node):
    def __init__(self):
        super().__init__('wit_imu_driver')
        
        # 参数配置
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.port = self.get_parameter('port').value
        
        # 发布器
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)
        
        # 串口初始化：增加错误捕获与保护
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=0.1)
            self.get_logger().info(f"✅ 传感器已就绪: {self.port}")
        except Exception as e:
            self.get_logger().error(f"❌ 串口开启失败: {e}")
            raise e

        self.buffer = bytearray()
        # 定时检查串口，保持高频读取防止缓冲区堆积
        self.timer = self.create_timer(0.002, self.read_data)

    def read_data(self):
        if self.ser.in_waiting > 0:
            self.buffer.extend(self.ser.read(self.ser.in_waiting))
        
        # 协议解析：寻找 55 61 包头，包长 20 字节
        while len(self.buffer) >= 20:
            if self.buffer[0] == 0x55 and self.buffer[1] == 0x61:
                packet = self.buffer[:20]
                self.process_imu(packet)
                del self.buffer[:20]
            else:
                self.buffer.pop(0)

    def process_imu(self, pkg):
        """解析协议并转换 ROS 标准单位"""
        # 16位有符号整数转换函数
        def to_int16(h, l):
            val = (h << 8) | l
            return val if val < 32768 else val - 65536

        # --- 数据提取 ---
        # 加速度: g -> m/s^2
        ax = to_int16(pkg[3], pkg[2]) / 32768.0 * 16.0 * 9.8
        ay = to_int16(pkg[5], pkg[4]) / 32768.0 * 16.0 * 9.8
        az = to_int16(pkg[7], pkg[6]) / 32768.0 * 16.0 * 9.8

        # 角速度: deg/s -> rad/s
        gx = math.radians(to_int16(pkg[9], pkg[8]) / 32768.0 * 2000.0)
        gy = math.radians(to_int16(pkg[11], pkg[10]) / 32768.0 * 2000.0)
        gz = math.radians(to_int16(pkg[13], pkg[12]) / 32768.0 * 2000.0)

        # 欧拉角: deg -> rad 姿态角解算时所使用的坐标系为东北天坐标系
        r = math.radians(to_int16(pkg[15], pkg[14]) / 32768.0 * 180.0)
        p = math.radians(to_int16(pkg[17], pkg[16]) / 32768.0 * 180.0)
        y = math.radians(to_int16(pkg[19], pkg[18]) / 32768.0 * 180.0)

        # --- 消息构建 ---
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        # 运动学数据
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = ax, ay, az
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gx, gy, gz

        # RPY 转换四元数 (ROS 标准坐标系计算) Roll->x Pitch->y Yaw->z
        #坐标系旋转顺序 定义为 Z-Y-X,即先绕 Z 轴转，再绕 Y 轴转，再绕 X 轴转。 
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy

        self.imu_pub.publish(msg)

    def destroy_node(self):
        # 显式关闭串口，防止重启时占用
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            # self.get_logger().info("🛑 串口连接已安全断开")
            print("🛑 串口连接已安全断开")
        super().destroy_node()

def main():
    rclpy.init()
    node = WitBleStandardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

if __name__ == '__main__':
    main()