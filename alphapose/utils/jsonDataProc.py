#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# __user__ = "wz"
import numpy as np
from numpy import ma
from pykalman import KalmanFilter
import json
import random
import cv2 as cv
import os

from .constantPath import *
from .vis_video import cal_IOU
from .rotate import judge_rotate

def data_kalman(angle_list):
    """
    利用卡尔曼滤波，用列表的前n个值去预测最后一个值
    线性数据能够取得较好的结果。
    :param 原始的angle_list list
    :return res: 返回预测的角度
    """
    angle_list = np.array(angle_list)
    masked = ma.masked_values(angle_list, -1)
    kf = KalmanFilter(initial_state_mean=masked.mean(), n_dim_obs=1)
    result = kf.em(masked).smooth(masked)
    target_list = []
    for item in result[0]:
        target_list.append(item[0])

    res = target_list
    return res


def data_kalman_1(angle_list):
    """
    利用卡尔曼滤波，用列表的前个值去预测第6个值
    :param 原始的angle_list list
    :return res: 返回预测的角度
    """
    kalmFil = cv.KalmanFilter(4, 2)
    kalmFil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalmFil.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalmFil.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003

    output = np.zeros_like(angle_list)
    for i, value in enumerate(angle_list):
        current_value = np.array([[np.float32(i)], [np.float32(value)]])
        kalmFil.correct(current_value)
        current_pred = kalmFil.predict()
        output[i] = current_pred[1].item()

    return int(output[-1])


def filling_data(angle_list):
    """
    填充角度列表中的-1。
    :param angle_list: 原始角度列表
    :return: result 填充后的角度
    """
    len_angle_list = len(angle_list)
    # 帧数太少， 直接返回
    if len_angle_list < 30:
        return angle_list

    for i in range(len_angle_list):
        # 如果有值不用填充，直接跳过循环。
        if angle_list[i] != -1:
            continue

        if i == 0:  # 第一个值比较关键
            if i + 1 < len_angle_list and angle_list[i + 1] != -1:
                angle_list[i] = (angle_list[i + 1] + 5) % 180 
            else:
                angle_list[i] = random.randint(0, 180)  # 随机赋值
        elif i == len_angle_list - 1:
            angle_list[i] = angle_list[i - 1]
        elif 0 < i < 10:
            if i + 1 < len_angle_list and angle_list[i + 1] != -1:
                angle_list[i] = (angle_list[i + 1] + 5) % 180
            else:
                angle_list[i] = (angle_list[i - 1] + 5) % 180
        elif 10 <= i < len_angle_list - 1:  # 根据前10个值预测当前值
            if angle_list[i - 1] != -1 and angle_list[i + 1] != -1:
                angle_list[i] = round(((angle_list[i - 1] + angle_list[i + 1]) / 2), 2)
            else:
                angle_list[i] = data_kalman_1(angle_list[i - 10: i])

    return angle_list 


# 废弃
def judge_jump(bbox):
    """
    判断人物是否跳跃。
    :param bbox: 每一帧的人物框，[[xmin, ymin, xmax, ymax], ...]
    :return jump: 每一帧是否跳跃，[bool, ...]
    """
    length = len(bbox)
    jump = [False for _ in range(length)]

    def judge(bboxs):
        """根据27个bbbox中心的偏移度在垂直方向上的偏移量判断发否跳跃"""
        assert len(bboxs) == 27, "the length of bboxs isn't 27"
        point = []  # 只关注垂直方向上的偏差
        for item in bboxs:
            if item[0] != -1:
                point.append(int((item[1] + item[3]) / 2))
            else:
                point.append(0)
        # 如果方差为零，证明所有元素相同，直接返回False
        point = np.array(point)
        if np.var(point) == 0:
            return False
        # 用非零元素的平均值填充0
        exist = (point != 0)
        mean_value = point.sum() / exist.sum()
        point[point == 0] = mean_value
        std = np.std(point)
        # 对std:标准差卡一个阈值来判断是否跳跃
        if std > 20:
            return True
        else:
            return False

    # 当前帧根据前后五帧的bbox的中心点偏移程度进行判断。
    for i in range(13, length - 13):
        jump[i] = judge(bbox[i - 13: i + 14])

    return jump


def get_angle_mean(angles):
    """
    返回角度相对于前后13帧的平均值。
    :param angles: list 所有角度。
    :return: 角度平均值 或 -1表示无效值。
    """
    length = len(angles)
    assert length >= 30, "the length of angle list is less than 30"
    angle_mean = [0 for _ in range(length)]

    def calc_mean(angle_list):
        """计算27个角度的平均值"""
        assert len(angle_list) == 27, "the length of bboxs isn't 27"
        angle_list = np.array(angle_list)
        if len(angle_list[angle_list != -1]) == 0:
            return -1
        else:
            return np.mean(angle_list[angle_list != -1])

    # 计算当前帧以及前后十三帧的角度平均值
    for i in range(13, length - 13):
        angle_mean[i] = calc_mean(angles[i - 13: i + 14])
    return angle_mean


def get_bbox_std(bboxs):
    """
    返回bbox中心点相对于前后13帧的标准差。
    :param bboxs: 每一帧的人物框，[[xmin, ymin, xmax, ymax], ...]
    :return bbox_std: 每一帧bbox的标准差，前后13帧默认为0.
    """
    length = len(bboxs)
    assert length >= 30, "the length of bboxs is less than 30"
    bbox_std = [0 for _ in range(length)]

    def judge(bboxs):
        """计算27个bbbox中心的偏移度在垂直方向上的偏移量"""
        assert len(bboxs) == 27, "the length of bboxs isn't 27"
        point = []  # 只关注垂直方向上的偏差
        for item in bboxs:
            if item[0] != -1:
                point.append(int((item[1] + item[3]) / 2))
            else:
                point.append(0)
        # 如果方差为零，证明所有元素相同，直接返回False
        point = np.array(point)
        if np.var(point) == 0:
            return False
        # 用非零元素的平均值填充0
        exist = (point != 0)
        mean_value = point.sum() / exist.sum()
        point[point == 0] = mean_value
        return np.std(point)

    # 计算当前帧以及前后十三帧的bbox的中心点偏移程度
    for i in range(13, length - 13):
        bbox_std[i] = judge(bboxs[i - 13: i + 14])

    return bbox_std

def get_angle_mean_1(angles):
    """
    返回角度相对于前后5帧的平均值。
    :param angles: list 所有角度。
    :return: 角度平均值 或 -1表示无效值。
    """
    length = len(angles)
    assert length >= 30, "the length of angle list is less than 30"
    angle_mean = [0 for _ in range(length)]

    def calc_mean(angle_list):
        """计算11个角度的平均值"""
        assert len(angle_list) == 11, "the length of bboxs isn't 11"
        angle_list = np.array(angle_list)
        if len(angle_list[angle_list != -1]) == 0:
            return -1
        else:
            return np.mean(angle_list[angle_list != -1])

    # 计算当前帧以及前后十三帧的角度平均值
    for i in range(5, length - 5):
        angle_mean[i] = calc_mean(angles[i - 5: i + 6])
    return angle_mean

def get_bbox_std_1(bboxs):
    """
    返回bbox中心点相对于前后5帧的标准差。
    :param bboxs: 每一帧的人物框，[[xmin, ymin, xmax, ymax], ...]
    :return bbox_std: 每一帧bbox的标准差，前后5帧默认为0.
    """
    length = len(bboxs)
    assert length >= 30, "the length of bboxs is less than 30"
    bbox_std = [0 for _ in range(length)]

    def judge(bboxs):
        """计算5个bbbox中心的偏移度在垂直方向上的偏移量"""
        assert len(bboxs) == 11, "the length of bboxs isn't 11"
        point = []  # 只关注垂直方向上的偏差
        for item in bboxs:
            if item[0] != -1:
                point.append(int((item[1] + item[3]) / 2))
            else:
                point.append(0)
        # 如果方差为零，证明所有元素相同，直接返回False
        point = np.array(point)
        if np.var(point) == 0:
            return False
        # 用非零元素的平均值填充0
        exist = (point != 0)
        mean_value = point.sum() / exist.sum()
        point[point == 0] = mean_value
        return np.std(point)

    # 计算当前帧以及前后十三帧的bbox的中心点偏移程度
    for i in range(5, length - 5):
        bbox_std[i] = judge(bboxs[i - 5: i + 6])

    return bbox_std


def calc_bbox(bbox0, bbox1):
    """
    输入两个框，返回对应的flag
    :return flag: 0表示不进行动作识别，1表示判断是否为跳跃， 2表示判断是否为托举
    """
    # 
    if bbox0[0] == -1 or bbox1[0] == -1:
        return 0
    _, iou = cal_IOU(bbox0, bbox1, 1)
    if iou > 0:
        return 2
    else:
        return 1

def jump_data_proc(jump):
    """
    ：param jump：list
    ：return jump_proc：list
    """
    length = len(jump)
    jump_proc = [False for _ in range(length)]
    assert len(jump_proc) == length
    # 长度太短直接返回
    if length < 20:
        return jump_proc

    for i in range(10, length - 10):
        tmp = jump[i - 9: i + 9]  
        num = tmp.count(True)  # 计算True的个数
        if num > 10:
            jump_proc[i] = True
        else:
            jump_proc[i] = False
    
    return jump_proc

def jump_datas_proc(jump0, jump1):
    """
    比对两个人同作，相互作为参考。
    """
    len0 = len(jump0)
    len1 = len(jump1)
    assert len0 == len1
    res0 = jump0.copy()
    res1 = jump1.copy()
    for i in range(20, len0 - 20):
        if jump0[i] == True and sum(jump1[i - 20 : i + 21]) == 0:
            res0[i] = False 

        if jump1[i] == True and sum(jump0[i - 20 : i + 21]) == 0:
            res1[i] = False      

    return res0, res1

def json_data_process():
    """
    对output_source.json中的数据进行处理。
    """
    # 读取原始数据
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        print("加载output_source.json文件。")


    # 动作判断
    item0 = info[0]
    len0 = len(item0['bbox'])
    bbox_std_0 = get_bbox_std(item0['bbox'])
    left_hip_std_0 = get_angle_mean(item0['angle']['left_hip'])
    right_hip_std_0 = get_angle_mean(item0['angle']['right_hip'])
    jump0 = [False for _ in range(len0)]
    lift0 = [False for _ in range(len0)]

    item1 = info[1]
    len1 = len(item1['bbox'])
    bbox_std_1 = get_bbox_std(item1['bbox'])
    left_hip_std_1 = get_angle_mean(item1['angle']['left_hip'])
    right_hip_std_1 = get_angle_mean(item1['angle']['right_hip'])
    jump1 = [False for _ in range(len1)]
    lift1 = [False for _ in range(len1)]
    assert len0 == len1, "error, len0 != len1."

    # 帧数太少，跳过处理
    if len0 > 30:
        for i in range(len0):
            bbox0 = item0['bbox'][i]
            bbox1 = item1['bbox'][i]
            # 根据bbox的相对位置来区分是进行跳跃判断还是托举判断
            # flag = calc_bbox(bbox0, bbox1)
            flag = 1
            # 跳跃两个人分开进行判断
            if bbox_std_0[i] > 25 and left_hip_std_0[i] > 145 and right_hip_std_0[i] > 145:
                jump0[i] = True

            if bbox_std_1[i] > 25 and left_hip_std_1[i] > 145 and right_hip_std_1[i] > 145:
                jump1[i] = True     

            # 托举废弃

            # bbox的iou不为零判断是否可能为托举
            # 托举两个人同时进行判断 lift1 == lift2
            # 计算两个bbox的水平和垂直距离
            # x0 = int((bbox0[0] + bbox0[2]) / 2)
            # y0 = int((bbox0[1] + bbox0[3]) / 2)
            # x1 = int((bbox1[0] + bbox1[2]) / 2)
            # y1 = int((bbox1[1] + bbox1[3]) / 2)
            # 卡一个阈值
            #if abs(y0 - y1) > abs(x0 - x1) and abs(x0 - x1) < 50:
                # 加一个手臂角度判断,被托举的人y更小
                #if y0 < y1 and (item1['angle']['left_shoulder'] > 145 or item1['angle']['right_shoulder'] > 145):  # y1是托举的人
                #    lift0[i] = True
                #    lift1[i] = True
                #elif y0 > y1 and (item0['angle']['left_shoulder'] > 145 or item0['angle']['right_shoulder'] > 145):  # y0是托举的人
                #    lift0[i] = True
                #    lift1[i] = True

    # jump处理 
    jump0 = jump_data_proc(jump0)
    jump1 = jump_data_proc(jump1)
    jump0, jump1 = jump_datas_proc(jump0, jump1)
    item0['action']['jump'] = jump0
    item0['action']['lift'] = lift0
    item1['action']['jump'] = jump1
    item1['action']['lift'] = lift1
    del item0, item1 
    
    # 数据处理
    for item in info:

        # 填充角度
        angle = item['angle']
        for key, value in angle.items():
            angle[key] = filling_data(value)
        # 删除bbox这一项
        del item['bbox']
        # del item['debug']

    with open(json_output_path, 'w') as fp:
        json.dump(info, fp)
        print("处理完毕，写入output.json文件。")

    # judge rotate
    judge_rotate()

    # # 清除文件
    # if os.path.exists(json_video_path):
    #     os.remove(json_video_path)

    if os.path.exists(json_source_path):
        os.remove(json_source_path)



if __name__ == '__main__':
    # json_data_process()
    pass
