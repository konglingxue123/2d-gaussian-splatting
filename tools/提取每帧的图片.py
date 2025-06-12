#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
# File      :提取每帧的图片.py.py
# Date      :2023/11/24 下午2:54
# Author    :konglx
"""
import cv2

def extract_frames(video_path, output_folder, frames_per_second=1):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # 计算每秒应该提取多少帧
    frames_to_extract = int(fps / frames_per_second)

    # 记录帧数
    frame_count = 0

    while True:
        # 读取视频的一帧
        ret, frame = video_capture.read()

        # 检查是否到达视频末尾
        if not ret:
            break

        # 每秒提取指定帧数
        if frame_count % frames_to_extract == 0:
            # 构造输出文件名
            output_filename = f"{output_folder}/frame_{frame_count // frames_to_extract:04d}.png"

            # 保存当前帧为图片
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(output_filename, frame)
            print('output_filename:', output_filename)

        frame_count += 1
        

    # 释放视频捕获对象
    video_capture.release()

if __name__ == "__main__":
    import os
    # 输入视频文件路径
    # video_path = "/home/ubunto/Project/konglx/seg/UniMatch-main/data/video/yongjiang/DJI_0679.MP4"
    video_path = '/home/ubunto/Project/konglx/pcd/3dgs/gaussian-splatting-main/datasets/北京昌平-百葛桥-钢管混凝土腐蚀/WeChat_20250507133211.mp4'

    # 输出文件夹路径
    # output_folder = "/home/ubunto/Project/konglx/seg/UniMatch-main/data/video/each_frame/679"
    output_folder = '/home/ubunto/Project/konglx/pcd/2dgs/2d-gaussian-splatting-main/datasets/beijing_baigeqiao'
    os.makedirs(output_folder, exist_ok=True)   
    # 指定每秒提取的帧数
    frames_per_second = 2

    # 提取帧数并保存为图片
    extract_frames(video_path, output_folder, frames_per_second)
