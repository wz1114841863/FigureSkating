# -*- coding: UTF-8 -*-
from __future__ import division
import cv2 as cv
import json
import av
from .constantPath import *

# FONT
DEFAULT_FONT = cv.FONT_HERSHEY_SIMPLEX
# COLOR
RED = (0, 0, 255)


def flag_draw(frame, location):
    """
    如果jump为Ture， 绘制跳跃标记
    :param frame: 原始图片
    :param location: 文本绘制的位置
    :return img: 返回绘制后的视频
    """
    img = frame.copy()
    cv.putText(img, "Jump", tuple(location), DEFAULT_FONT, 1, RED, 1, cv.LINE_AA)
    return img


def jump_draw():
    # 获取视频
    capture = cv.VideoCapture(video_input_path)
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    # 创建保存路径和视频索引
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    video_result = cv.VideoWriter('./output_jump.mp4', fourcc, 26, (frame_width, frame_height), True)
    # 读取output.json
    with open(json_output_path, 'r') as fp:
        infos = json.load(fp)

    # 依次处理
    length = len(infos[0]['jump'])
    for i in range(length):
        ret, img = capture.read()
        if not ret:
            print("读取帧失败， 函数结束。")
            break

        # 读取对应注释文件
        jump_0 = infos[0]['jump'][i]
        jump_1 = infos[1]['jump'][i]
        height, width = img.shape[:2]
        if jump_0:
            img = flag_draw(img, [0 + 50, 0 + 50])

        if jump_1:
            img = flag_draw(img, [0 + 50, height - 50])

        video_result.write(img)
    # 释放视频索引
    capture.release()
    video_result.release()


def read_and_save():
    """单纯的读取并保存"""
    # 获取视频
    capture = cv.VideoCapture(video_input_path)
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_fps = 26
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    # 创建保存路径和视频索引
    container = av.open(video_output_path, mode='w')
    stream = container.add_stream('h264', rate=frame_fps)
    stream.width = frame_width
    stream.height = frame_height
    stream.pix_fmt = 'yuv420p'
    for i in range(frame_count):
        # 读取帧
        ret, img = capture.read()
        if not ret:
            break
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')

        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()
    capture.release()


def read_and_save_1():
    """单纯的读取并保存"""
    # 获取视频
    capture = cv.VideoCapture(video_input_path)
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'FMP4')  # .avi
    video_result = cv.VideoWriter(video_output_path, fourcc, 26, (frame_width, frame_height), True)
    # 逐帧读取图像和注释
    for idx in range(frame_count):
        # 读取帧
        ret, img = capture.read()
        if not ret:
            print("读取帧失败， 函数结束。")
            break
        video_result.write(img)
    # 释放视频索引
    capture.release()
    video_result.release()


if __name__ == '__main__':
    jump_draw()
    # read_and_save_1()
