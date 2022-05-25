#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from numpy import ma
from pykalman import KalmanFilter
import json

json_source_path = "/home/dell/first/AlphaPose/examples/demo/output/output_source.json"
json_output_path = "/home/dell/first/AlphaPose/examples/demo/output/output.json"


def data_kalman(angle_list):
    """
    对每一个角度进行卡尔曼滤波。
    线性数据能够取得较好的结果。
    :param 原始的angle_list list
    :return: 处理后的target_list list
    """
    angle_list = np.array(angle_list)
    masked = ma.masked_values(angle_list, -1)
    kf = KalmanFilter(initial_state_mean=masked.mean(), n_dim_obs=1)
    result = kf.em(masked).smooth(masked)
    target_list = []
    for item in result[0]:
        target_list.append(item[0])

    return target_list


def filling_data(angle_list):
    """
    填充角度列表中的-1。
    :param angle_list: 原始角度 列表
    :return: result 填充后的角度列表
    """
    masked = ma.masked_values(angle_list, -1)
    mean_angle = int(masked.mean())
    result = angle_list
    len_angle_list = len(angle_list)
    for i in range(len_angle_list):
        if angle_list[i] != -1:
            continue
        if i == 0:
            if angle_list[i + 1] != -1:
                result[i] = angle_list[i + 1]
            else:
                result[i] = mean_angle
        elif i == len_angle_list - 1:
            result[i] = angle_list[i - 1]
        else:
            if angle_list[i - 1] != -1 and angle_list[i + 1] != -1:
                result[i] = int((angle_list[i - 1] + angle_list[i + 1]) / 2)
            elif angle_list[i + 1] == -1:
                result[i] = angle_list[i - 1]

    return result


def json_data_process():
    """
    对output_source.json中的数据进行处理。
    :return: None
    """
    # 读取原始数据
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        print("加载output_source.json文件。")
        print(type(info))
    for item in info:
        angle = item['angle']
        for key, value in angle.items():
            # angle[key] = data_kalman(value)
            angle[key] = filling_data(value)

    with open(json_output_path, 'w') as fp:
        json.dump(info, fp)
        print("处理完毕，写入output.json文件。")


if __name__ == '__main__':
    json_data_process()
