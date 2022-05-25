# -*- coding: UTF-8 -*-
"""视频后处理"""
from __future__ import division
import cv2 as cv
import json
from .vis_video import vis_frame_empty
from .vis_video import vis_frame
from .jsonDataProc import json_data_process
import av
import numpy as np
from .constantPath import *
import os


def video_post_proc(data, path):
    """
    视频后处理，按帧读取视频和注释文档，绘制后写入文件。
    利用opencv实现视频保存。
    :param data:输入信息
    :return:
    """
    video_input_path = path
    capture = cv.VideoCapture(video_input_path)
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_fps = int(capture.get(cv.CAP_PROP_FPS))
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print("视频获取失败。")
    else:
        print("视频获取成功。")
        print(f"视频总帧数：{frame_count}")
        print(f"视频高度：{frame_height}")
        print(f"视频宽度数：{frame_width}")
        print(f"视频帧率：{frame_fps}")
    # 创建保存路径和视频索引
    # fourcc = cv.VideoWriter_fourcc(*'XVID')  # .avi
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv.VideoWriter_fourcc(*'FMP4')  # .avi

    video_result = cv.VideoWriter(video_output_tmp_path, fourcc, frame_fps, (frame_width, frame_height), True)
    # 处理data
    data_0 = data['0']
    data_1 = data['1']
    # 添加到output.json
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print("读取output_source.json文件。")

    for item in info:
        if item['idx'] == 1:
            item['name'] = data_0['name']
            item['sex'] = data_0['sex']
        elif item['idx'] == 2:
            item['name'] = data_1['name']
            item['sex'] = data_1['sex']
        else:
            print("error, Unknowed id.")

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")

    del info
    # 读取json文件
    with open(json_video_path, 'r') as fp:
        annos = json.load(fp)
    len_annos = len(annos)
    print(f"len(annos):{len_annos}")
    # assert len_annos == frame_count, "视频帧数和注释长度不符合"
    # 逐帧读取图像和注释
    for idx in range(len_annos):  # n
        # 读取帧
        ret, img = capture.read()
        if not ret:
            print("读取帧失败， 函数结束。")
            break
        # 降低亮度
        img = np.uint8(np.clip((img * 0.80), 0, 255))
        # 读取对应注释文件
        anno = annos[idx]
        len_anno = len(anno['result'])
        if len_anno == 0:
            # print(f"the {idx}th image dosen't has annos.")
            img = vis_frame_empty(img)
        elif len_anno == 1:
            if anno['result'][0]['idx'] == 1:  # id == 1
                anno['result'][0]['name'] = data_0['name']
                anno['result'][0]['sex'] = data_0['sex']
            else:
                anno['result'][0]['name'] = data_1['name']
                anno['result'][0]['sex'] = data_1['sex']

            img = vis_frame(img, anno)
        elif len_anno == 2:
            if anno['result'][0]['idx'] == 1:  # id == 1
                anno['result'][0]['name'] = data_0['name']
                anno['result'][0]['sex'] = data_0['sex']
                anno['result'][1]['name'] = data_1['name']
                anno['result'][1]['sex'] = data_1['sex']
            else:
                anno['result'][0]['name'] = data_1['name']
                anno['result'][0]['sex'] = data_1['sex']
                anno['result'][1]['name'] = data_0['name']
                anno['result'][1]['sex'] = data_0['sex']
            img = vis_frame(img, anno)
        else:
            text = "the length of annos beyond expection."
            raise ValueError(text)

        # 写入文件
        video_result.write(img)
    # 调用jsonDataProc函数
    json_data_process()
    # 释放视频索引
    capture.release()
    video_result.release()
    # 调用ffmpeg修改视频编码格式
    ffmp = "ffmpeg -i " + video_output_tmp_path + " -b:v 5M -vcodec libx264 -y -v quiet " + video_output_path
    os.system(ffmp)
    # if os.path.exists(video_output_tmp_path):
    #     os.remove(video_output_tmp_path)

    # if os.path.exists(video_intput_path):
    #     os.remove(video_intput_path)


def video_post_proc_1(data, path=""):
    """
    视频后处理，按帧读取视频和注释文档，绘制后写入文件。
    利用pyav实现视频保存
    :return:
    """
    # 输入视频
    video_input_path = path
    capture = cv.VideoCapture(video_input_path)
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_fps = int(cv.CAP_PROP_FPS)
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print("视频获取失败。")
    else:
        print("视频获取成功。")
        print(f"视频总帧数：{frame_count}")
        print(f"视频高度：{frame_height}")
        print(f"视频宽度数：{frame_width}")
        print(f"视频帧率：{frame_fps}")
    # 处理data
    data_0 = data['0']
    data_1 = data['1']
    # 添加到output.json
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print("读取output_source.json文件。")

    for item in info:
        if item['idx'] == 1:
            item['name'] = data_0['name']
            item['sex'] = data_0['sex']
        elif item['idx'] == 2:
            item['name'] = data_1['name']
            item['sex'] = data_1['sex']
        else:
            print("error, Unknowed id.")

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")

    del info
    # 创建保存路径和视频索引
    container = av.open(video_output_path, mode='w')
    stream = container.add_stream('h264', rate=26)
    stream.width = frame_width
    stream.height = frame_height
    # stream.pix_fmt = 'yuv444p'
    stream.pix_fmt = 'nv12'
    # 读取json文件
    with open(json_video_path, 'r') as fp:
        annos = json.load(fp)
    len_annos = len(annos)
    print(f"len(annos):{len_annos}")
    # assert len_annos == frame_count, "视频帧数和注释长度不符合"
    # 逐帧读取图像和注释
    for idx in range(len_annos):
        # 读取帧
        ret, img = capture.read()
        if not ret:
            print("读取帧失败， 函数结束。")
            break
        # 读取对应注释文件
        anno = annos[idx]
        len_anno = len(anno['result'])
        if len_anno == 0:
            # print(f"the {idx}th image dosen't has annos.")
            img = vis_frame_empty(img)
        elif len_anno == 1:
            if anno['result'][0]['idx'] == 1:  # id == 1
                anno['result'][0]['name'] = data_0['name']
                anno['result'][0]['sex'] = data_0['sex']
            else:
                anno['result'][0]['name'] = data_1['name']
                anno['result'][0]['sex'] = data_1['sex']

            img = vis_frame(img, anno)
        elif len_anno == 2:
            if anno['result'][0]['idx'] == 1:  # id == 1
                anno['result'][0]['name'] = data_0['name']
                anno['result'][0]['sex'] = data_0['sex']
                anno['result'][1]['name'] = data_1['name']
                anno['result'][1]['sex'] = data_1['sex']
            else:
                anno['result'][0]['name'] = data_1['name']
                anno['result'][0]['sex'] = data_1['sex']
                anno['result'][1]['name'] = data_0['name']
                anno['result'][1]['sex'] = data_0['sex']
            img = vis_frame(img, anno)
        else:
            text = "the length of annos beyond expection."
            raise ValueError(text)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(img, format='rgb24')

        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    # 调用jsonDataProc函数
    json_data_process()
    # 释放视频索引
    capture.release()
    container.close()


if __name__ == '__main__':
    # video_post_proc()
    import torch

    print(torch.cuda.is_available())
