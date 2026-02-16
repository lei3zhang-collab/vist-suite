#!/usr/bin/env python3
"""
Astra Pro Stereo Calibration Engine: RGB and Infrared Sensor Alignment

This script provides a comprehensive solution for calibrating the intrinsic and 
extrinsic parameters of the Astra Pro stereo camera system. It processes 
synchronized RGB and IR image pairs to compute camera matrices, distortion 
coefficients, and the spatial transformation (Rotation and Translation) between 
the two sensors. Built-in histogram equalization is applied to the IR stream 
to enhance chessboard corner detection in low-contrast environments.

功能概述 (Functionality Overview):
-------------------------------
1. 单目标定 (Intrinsic): 独立解算彩色(RGB)与红外(IR)相机的内参矩阵与畸变参数。
2. 双目标定 (Extrinsic): 基于同步角点对，计算 RGB 相对于 IR 的位置关系 (R, T)。
3. IR 图像增强: 自动对 IR 图像进行直方图均衡化，极大提升了角点识别的鲁棒性。
4. 亚像素精化: 采用 cornerSubPix 技术，确保角点定位精度达到亚像素级别。
5. 结果导出: 自动生成 YAML 格式的标定文件，可直接用于 ROS 2 深度图注册 (Registration)。

可配置参数 (Configurable Parameters):
-------------------------------------
1. SQUARE_SIZE (float, default: 23.85): 棋盘格单个方格的物理边长 (mm)。
2. CHECKERBOARD (tuple, default: (9, 6)): 棋盘格内部角点的数量 (列, 行)。
3. CALIBRATE_STEREO (bool): 是否执行双目外参解算。

路径规范 (Path Conventions):
---------------------------
- 输入路径: ~/workspace_Ubuntu/data/calib_raw/ (需包含 rgb_*.png 和 ir_*.png)
- 输出路径: ~/workspace_Ubuntu/data/calibration_params/calibration_params.yaml

作者: Zhang Lei
日期: 2026-02-14
"""


import cv2
import numpy as np
import glob
import yaml
import os
# ==========================================
# 工程配置开关
# ==========================================
SQUARE_SIZE = 23.85      # 格子边长 (mm)
CALIBRATE_IR = True     # 是否解算 IR 单目标定
CALIBRATE_STEREO = True # 是否解算双目外参 (前提是有同步角点)
# ==========================================
def main():
    CHECKERBOARD = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备 3D 世界坐标
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 容器初始化
    obj_points_rgb, img_points_rgb = [], []
    obj_points_ir, img_points_ir = [], []
    obj_points_stereo, img_points_rgb_stereo, img_points_ir_stereo = [], [], []


    rgb_images = sorted(glob.glob('./data/calib_raw/rgb_*.png'))
    ir_images = sorted(glob.glob('./data/calib_raw/ir_*.png'))
    img_shape = None

    print(f"检测到图像对数量: {len(rgb_images)}")

    for rgb_path, ir_path in zip(rgb_images, ir_images):
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path, 0)
        if img_shape is None: img_shape = img_rgb.shape[1::-1]

        # RGB 检测
        ret_rgb, corners_rgb = cv2.findChessboardCorners(img_rgb, CHECKERBOARD, None)
        c_rgb2 = None
        if ret_rgb:
            obj_points_rgb.append(objp)
            c_rgb2 = cv2.cornerSubPix(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), corners_rgb, (11,11), (-1,-1), criteria)
            img_points_rgb.append(c_rgb2)

        # IR 检测 (仅当开关开启或需要双目时运行)
        ret_ir = False
        c_ir2 = None
        if CALIBRATE_IR or CALIBRATE_STEREO:
            ir_enhanced = cv2.equalizeHist(img_ir)
            ret_ir, corners_ir = cv2.findChessboardCorners(ir_enhanced, CHECKERBOARD, None)
            if ret_ir:
                obj_points_ir.append(objp)
                c_ir2 = cv2.cornerSubPix(ir_enhanced, corners_ir, (11,11), (-1,-1), criteria)
                img_points_ir.append(c_ir2)

        # 双目同步判断
        if ret_rgb and ret_ir:
            obj_points_stereo.append(objp)
            img_points_rgb_stereo.append(c_rgb2)
            img_points_ir_stereo.append(c_ir2)
            status = " [BOTH OK]"
        else:
            status = f" [RGB:{'OK' if ret_rgb else 'FAIL'} | IR:{'OK' if ret_ir else 'FAIL'}]"
        print(f"处理: {os.path.basename(rgb_path)}{status}")

    # --- 开始解算 ---
    result = {}

    # 1. RGB 必算
    if len(img_points_rgb) > 5:
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(obj_points_rgb, img_points_rgb, img_shape, None, None)
        result.update({'K_rgb': mtx_r.tolist(), 'D_rgb': dist_r.tolist(), 'rms_rgb': float(ret_r)})
        print(f"\n✅ RGB 标定完成, RMS: {ret_r:.4f}")

    # 2. IR 可选
    if CALIBRATE_IR and len(img_points_ir) > 5:
        ret_i, mtx_i, dist_i, _, _ = cv2.calibrateCamera(obj_points_ir, img_points_ir, img_shape, None, None)
        result.update({'K_ir': mtx_i.tolist(), 'D_ir': dist_i.tolist(), 'rms_ir': float(ret_i)})
        print(f"✅ IR 标定完成, RMS: {ret_i:.4f}")

    # 3. 双目可选
    if CALIBRATE_STEREO and len(obj_points_stereo) > 0:
        if 'K_rgb' in result and 'K_ir' in result:
            print(f"🚀 开始双目外参计算 (样本数: {len(obj_points_stereo)})...")
            # ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            #     obj_points_stereo, img_points_rgb_stereo, img_points_ir_stereo,
            #     np.array(result['K_rgb']), np.array(result['D_rgb']),
            #     np.array(result['K_ir']), np.array(result['D_ir']),
            #     img_shape, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5), 
            #     flags=cv2.CALIB_FIX_INTRINSIC)
            ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                obj_points_stereo, img_points_ir_stereo, img_points_rgb_stereo,  # 交换顺序 RGB是base，R T 计算的是 IR 相对于 RGB 的变换，也就是 IR在 RGB 坐标系下的位置
                np.array(result['K_ir']), np.array(result['D_ir']),              # IR 内参作为第一个
                np.array(result['K_rgb']), np.array(result['D_rgb']),            # RGB 内参作为第二个
                img_shape, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5), 
                flags=cv2.CALIB_FIX_INTRINSIC)
            result.update({'R': R.tolist(), 'T': T.tolist(), 'rms_stereo': float(ret_s)})
            print(f"✅ 双目对齐完成, RMS: {ret_s:.4f}")
        else:
            print("❌ 错误: 双目标定需要 RGB 和 IR 的内参都解算成功！")

    # 保存结果
    if result:
        with open('./data/calibration_params/calibration_params.yaml', 'w') as f:
            yaml.dump(result, f)
        print("\n配置文件已保存: calibration_params.yaml")

if __name__ == '__main__':
    main()
    