import math
import time

import cv2 as cv
import numpy as np
import torch

import json

# COLOR BGR
CYAN = (226, 215, 8)
LIGHT_BLUE = (255, 108, 4)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE_1 = (255, 191, 0)
BLUE_2 = (255, 153, 102)

# FONT
DEFAULT_FONT = cv.FONT_HERSHEY_SIMPLEX

# annotations：the number of keypoints is 17
limbPairs = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),  # Body
             (11, 13), (12, 14), (13, 15), (14, 16)]  # Foot
"""
limbPairsColor = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),  # 未使用
                  WHITE, BLUE_2, BLUE_2, ORANGE, ORANGE, BLUE_2, ORANGE,
                  BLUE_2, ORANGE, BLUE_2, ORANGE]  # 左右两边使用不一样的颜色
"""
limbPairsColor = [CYAN for _ in range(15)]

keyPointsNames = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]

FLAG = True
# 每隔五帧刷新一次文本
count_frame_rate = 0
angle = {
    "id": 0.0,
    "name": "",
    "sex": "",
    "left_shoulder": 0.0,
    "right_shoulder": 0.0,
    "left_hip": 0.0,
    "right_hip": 0.0,
    "left_knee": 0.0,
    "right_knee": 0.0
}
angle_info = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee"]

json_file_path = "/home/dell/first/AlphaPose/examples/demo/output/output_source.json"


# 绘图函数
def img_show(img, windowName='image', waitSec=0):
    """
    显示图片。
    :param img: 原始图片。
    :param windowName: 窗口命名。
    :param waitSec: 等待的时间。
    :return: None
    """
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.imshow(windowName, img)
    idKey = cv.waitKey(waitSec)
    if idKey == 27:  # 27 为 ESC 键对应的 ASCII 码
        # 关闭指定窗口
        cv.destroyWindow(windowName)


def img_draw(img, personAnno=None):
    """
    绘制两个人物的框和关键点。
    :param img: 原始图片。
    :param personAnno: 标注数据。
    :return: None
    """
    if personAnno is None:
        return
    bbox_draw(img, personAnno[0]["bbox"], personAnno[1]["bbox"])
    for i in range(2):
        keypoints_draw(img, personAnno[i]["keypoints"])
        limbs_draw(img, personAnno[i]["keypoints"])


def calc_kp_angle(coordinates):
    """
    计算出角BAC的角度
    :param coordinates: 三个关节点列表[[xb, yb], [xa, ya], [xc, yc]]
    :return: 计算出的角度或者-1
    """
    assert len(coordinates) == 3
    coord_B, coord_A, coord_C = coordinates
    # 检查数据的有效性
    if coord_A[0] == -1 or coord_B[0] == -1 or coord_C[0] == -1:
        return -1
    # 计算距离
    distAB = math.sqrt((coord_A[0] - coord_B[0]) ** 2 + (coord_A[1] - coord_B[1]) ** 2)
    distAC = math.sqrt((coord_A[0] - coord_C[0]) ** 2 + (coord_A[1] - coord_C[1]) ** 2)
    distBC = math.sqrt((coord_B[0] - coord_C[0]) ** 2 + (coord_B[1] - coord_C[1]) ** 2)
    # 距离有效性
    if distAB == 0 or distAC == 0 or distBC == 0:
        return -1
    # cosA = [b²＋c²－a²] / (2bc)
    cosA = (distAB ** 2 + distAC ** 2 - distBC ** 2) / (2 * distAB * distAC)
    if cosA > 1 or cosA < -1:
        return -1
    angle_A = math.degrees(math.acos(cosA))
    angle_A = int(angle_A)
    return angle_A


# 暂时弃用
def single_kp_draw(img, keypoint):
    """
    绘制某个强调的关节点
    :param img: 原始图片
    :param keypoint: 强调关节点 [x1, y1]
    :return: None
    """
    # 检验关节点的有效性
    if keypoint[0] == -1:
        return None
    keypoint = tuple(keypoint)
    cv.circle(img, keypoint, 10, RED, -1)
    cv.circle(img, keypoint, 8, CYAN, -1)
    cv.circle(img, keypoint, 6, WHITE, -1)


def text_bbox_draw(img, text, location):
    """
    绘制角度信息
    :param img 原始图片
    :param text 要绘制的文本
    :param location 要绘制文本的位置 [x, y]
    :return: None
    """
    cv.putText(img, text, tuple(location), DEFAULT_FONT, 0.5, RED, 1, cv.LINE_AA)


def text_draw(img, keypoint, text):
    """
    绘制文本，描述关节点的角度。
    :param img: 原始图片
    :param keypoint: 要绘制的关键节点
    :param text: 要绘制的文本
    :return: None
    """
    # 检验关节点的有效性
    if keypoint[0] == -1:
        return

    cv.line(img, tuple(keypoint), (keypoint[0] - 50, keypoint[1] - 25), WHITE, 2)
    cv.line(img, (keypoint[0] - 50, keypoint[1] - 25), (keypoint[0] - 100, keypoint[1] - 25), WHITE, 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, text, (keypoint[0] - 100, keypoint[1] - 25), font, 0.5, RED, 2, cv.LINE_AA)


def limbs_draw(img, keypoints):
    """
    绘制关键点之间的连线。
    :param img: 原始图片
    :param keypoints: 绘制所有存在的关节对。[[x1, y1], [x2, y2], ...]
    :return: None
    """
    for i, pair in enumerate(limbPairs[4:]):  # 不绘制脸上的线
        if keypoints[pair[0]][0] != -1 and keypoints[pair[1]][0] != -1:
            cv.line(img, tuple(keypoints[pair[0]]), tuple(keypoints[pair[1]]), limbPairsColor[i + 4], 2)


def keypoints_draw(img, keypoints, color=ORANGE):
    """
    绘制两个人的关键点
    :param img: 原始图片
    :param keypoints: 人体关键点 [[x1, y1], [x2, y2], ...]
    :param color: 关键点的颜色
    :return: None
    """
    # 首先绘制所有可能的关键点
    for kp in keypoints[5:]:  # 不绘制脸部的点
        if kp[0] != -1:  # 关键点坐标存在
            cv.circle(img, tuple(kp), 4, color, -1)


def single_bbox_draw(_img, _bbox, _color=BLUE_1):
    """
    绘制一个人的bbo以及角线
    :param _img  原始图片
    :param _bbox  bbox  [xmin, ymin, xmax, ymax]
    :param _color 默认颜色
    """
    cv.rectangle(_img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), _color, 1)
    # 画直线来填充边框
    _color_1 = CYAN
    _color_2 = WHITE
    thickness_1 = 3
    thickness_2 = 2

    _bbox_WIDTH = _bbox[2] - _bbox[0]
    _bbox_HIGH = _bbox[3] - _bbox[1]
    length = int(min(_bbox_HIGH, _bbox_WIDTH) * 0.1)  # 统一角线的长度
    # 第一层
    # (xmin, ymin) --> right
    cv.line(_img, (_bbox[0], _bbox[1]), (_bbox[0] + length, _bbox[1]), _color_1, thickness_1)
    # (xmin, ymin) --> down
    cv.line(_img, (_bbox[0], _bbox[1]), (_bbox[0], _bbox[1] + length), _color_1, thickness_1)
    # (xmax, ymin) --> left
    cv.line(_img, (_bbox[2], _bbox[1]), (_bbox[2] - length, _bbox[1]), _color_1, thickness_1)
    # (xmax. xmin) --> down
    cv.line(_img, (_bbox[2], _bbox[1]), (_bbox[2], _bbox[1] + length), _color_1, thickness_1)
    # (xmin, ymax) --> right
    cv.line(_img, (_bbox[0], _bbox[3]), (_bbox[0] + length, _bbox[3]), _color_1, thickness_1)
    # (xmin, ymax) --> up
    cv.line(_img, (_bbox[0], _bbox[3]), (_bbox[0], _bbox[3] - length), _color_1, thickness_1)
    # (xmax  ymax) --> left
    cv.line(_img, (_bbox[2], _bbox[3]), (_bbox[2] - length, _bbox[3]), _color_1, thickness_1)
    # (xmax  ymax) --> up
    cv.line(_img, (_bbox[2], _bbox[3]), (_bbox[2], _bbox[3] - length), _color_1, thickness_1)
    # 第二层
    # (xmin, ymin) --> right
    cv.line(_img, (_bbox[0], _bbox[1]), (_bbox[0] + length, _bbox[1]), _color_2, thickness_2)
    # (xmin, ymin) --> down
    cv.line(_img, (_bbox[0], _bbox[1]), (_bbox[0], _bbox[1] + length), _color_2, thickness_2)
    # (xmax, ymin) --> left
    cv.line(_img, (_bbox[2], _bbox[1]), (_bbox[2] - length, _bbox[1]), _color_2, thickness_2)
    # (xmax. xmin) --> down
    cv.line(_img, (_bbox[2], _bbox[1]), (_bbox[2], _bbox[1] + length), _color_2, thickness_2)
    # (xmin, ymax) --> right
    cv.line(_img, (_bbox[0], _bbox[3]), (_bbox[0] + length, _bbox[3]), _color_2, thickness_2)
    # (xmin, ymax) --> up
    cv.line(_img, (_bbox[0], _bbox[3]), (_bbox[0], _bbox[3] - length), _color_2, thickness_2)
    # (xmax  ymax) --> left
    cv.line(_img, (_bbox[2], _bbox[3]), (_bbox[2] - length, _bbox[3]), _color_2, thickness_2)
    # (xmax  ymax) --> up
    cv.line(_img, (_bbox[2], _bbox[3]), (_bbox[2], _bbox[3] - length), _color_2, thickness_2)


def bbox_draw(img, bbox1, bbox2, color=BLUE_2):
    """
    绘制两人是bbox
    :param img: 原始图片
    :param bbox1: 第一个的人bbox [xmin, ymin, xmax, ymax]
    :param bbox2: 第二个人的bbox [xmin, ymin, xmax, ymax]
    :param color: bbox的颜色
    :return: None
    """
    # 计算两个框的iou，如果为零，分开绘制两个bbox，否则绘制在一起
    _, iou = cal_IOU(bbox1, bbox2, 1)
    if iou == 0:
        single_bbox_draw(img, bbox1, color)
        single_bbox_draw(img, bbox2, color)
    else:
        xmin = min(bbox1[0], bbox2[0])
        ymin = min(bbox1[1], bbox2[1])
        xmax = max(bbox1[2], bbox2[2])
        ymax = max(bbox1[3], bbox2[3])
        bbox = [xmin, ymin, xmax, ymax]
        single_bbox_draw(img, bbox)


def cal_IOU(bbox1, bbox2, flag=0):
    """
        bbox中提供的坐标： 左上 + 右下
    :param bbox1: [xleft1, yleft1, xright1, yright1]
        第一个bbox，坐标满足COCO坐标格式
    :param bbox2: [xleft2, yleft2, xright2, yright2]
        第二个bbox，坐标满足COCO坐标格式
    :param flag: 输入的bbox数据格式
        0：[[xmin, ymin], [xmax, ymax]]  1：[xmin, ymin, xmax, ymax]
    :return:
        inter_bbox: [xleft, yleft, xright, yright] inter_boox 的坐标用于可视化（如果存在的话）
        iou：float 返回两个框的置信度
    """
    # 统一格式
    if flag == 0:
        xmin1, ymin1, xmax1, ymax1 = bbox1[0][0], bbox1[0][1], bbox1[1][0], bbox1[1][1]
        xmin2, ymin2, xmax2, ymax2 = bbox2[0][0], bbox2[0][1], bbox2[1][0], bbox2[1][1]
    elif flag == 1:
        xmin1, ymin1, xmax1, ymax1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        xmin2, ymin2, xmax2, ymax2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    else:
        print("error。 the format is incorrect. \n")
        return None
    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    # 计算交点的坐标（xleft, yleft, xright, yright)
    xmin = np.max([xmin1, xmin2])
    ymin = np.max([ymin1, ymin2])
    xmax = np.min([xmax1, xmax2])
    ymax = np.min([ymax1, ymax2])
    inter_bbox = [xmin, ymin, xmax, ymax]
    # 计算交集面积
    inter_area = (np.max([0, xmax - xmin])) * (np.max([0, ymax - ymin]))  # 可能没有交集，此时面积为0
    # 计算iou
    # A = area(bbox1), B = area(bbox2), C = （area(bbox1) ∩ area(bbox2）
    # iou = C / (A + B - C)
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 防止分母为零

    return inter_bbox, iou


def empty_txt_draw(img):
    """
    在img的两边绘制空的文本。
    :param img: 原始图片
    :return: None
    """
    global angle
    height, width = img.shape[:2]
    # 绘制两个文本
    loc1 = [0 + 5, 0]
    loc2 = [width - 160, 0]
    offset = 15
    for key in angle:
        for item in (loc1, loc2):
            text_bbox_draw(img, key + "--", [item[0] + 5, item[1] + offset])
        offset += 15


# 原始保留--------------------
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval


def vis_frame_fast(frame, im_res, debug, opt, format='coco'):
    '''
    frame: frame image
    im_res: im_res of predictions
    :param frame_count
    format: coco or mpii
    debug：
    return rendered image
    '''
    kp_num = 17
    if len(im_res['result']) > 0:
        kp_num = len(im_res['result'][0]['keypoints'])
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                       # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                       # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                       (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    elif kp_num == 136:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
            (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
            (36, 37), (37, 38),
            # Face
            (38, 39), (39, 40), (40, 41), (41, 42), (43, 44), (44, 45), (45, 46), (46, 47), (48, 49), (49, 50),
            (50, 51), (51, 52),
            # Face
            (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61), (62, 63), (63, 64), (64, 65),
            (65, 66), (66, 67),
            # Face
            (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79),
            (79, 80), (80, 81),
            # Face
            (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91),
            (91, 92), (92, 93),
            # Face
            (94, 95), (95, 96), (96, 97), (97, 98), (94, 99), (99, 100), (100, 101), (101, 102), (94, 103), (103, 104),
            (104, 105),
            # LeftHand
            (105, 106), (94, 107), (107, 108), (108, 109), (109, 110), (94, 111), (111, 112), (112, 113), (113, 114),
            # LeftHand
            (115, 116), (116, 117), (117, 118), (118, 119), (115, 120), (120, 121), (121, 122), (122, 123), (115, 124),
            (124, 125),
            # RightHand
            (125, 126), (126, 127), (115, 128), (128, 129), (129, 130), (130, 131), (115, 132), (132, 133), (133, 134),
            (134, 135)
            # RightHand
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    elif kp_num == 26:
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
            (17, 18), (18, 19), (19, 11), (19, 12),
            (11, 13), (12, 14), (13, 15), (14, 16),
            (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                   (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                   (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)]  # foot

        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        if kp_num == 17:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        if opt.pose_track or opt.tracking:
            color = get_color_fast(int(abs(human['idx'])))
        else:
            color = BLUE

        # Draw bboxes
        if opt.showbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]]  # xmin,xmax,ymin,ymax
            else:
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)

            cv.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            if opt.tracking:
                cv.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, BLACK, 2)
        # Draw keypoints
        vis_thres = 0.05 if kp_num == 136 else 0.4
        for n in range(kp_scores.shape[0]):

            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                if opt.tracking:
                    cv.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if opt.tracking:
                        cv.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv.line(img, start_xy, end_xy, line_color[i],
                                2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv.line(img, start_xy, end_xy, (255, 255, 255), 1)

    cv.putText(img, im_res['imgname'] + str(debug), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return img


def vis_frame_empty(frame, frame_count):
    """
    没有预测到的帧，绘制文本框
    :param frame: 输入的原始图片
    :param frame_count: 图片属于视频的第几帧
    :return: img
    """
    # 获取图片属性
    img = frame.copy()
    height, width = img.shape[:2]
    # imgname = im_res['imgname']
    # 引入全局变量
    # 文本区域加一层灰色透明图
    background_left = np.zeros((150, 155, 3), dtype=np.uint8) + 125  # 灰色
    background_right = np.zeros((150, 160, 3), dtype=np.uint8) + 125  # 灰色
    roiLeft = img[0:150, 0:155]
    roiRight = img[0:150, width - 160:width]
    img[0:150, 0:155] = cv.addWeighted(background_left, 0.5, roiLeft, 0.5, 0)
    img[0:150, width - 160:width] = cv.addWeighted(background_right, 0.5, roiRight, 0.5, 0)
    del background_left, background_right, roiLeft, roiRight
    empty_txt_draw(img)
    # print(f"视频的第{frame_count}帧")
    # 读取json文件并追加
    with open(json_file_path, 'r') as fp:
        info = json.load(fp)
        # print("读取output.json文件。")
    # 追加 -1
    global angle_info
    for item in info:
        for angle_name in angle_info:
            item['angle'][angle_name].append(-1)
        item['jump'].append(False)
        bbox = [-1, -1, -1, -1]  # xmin, ymin, xmax, ymax
        item['bbox'].append(bbox)

    with open(json_file_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output.json文件。")

    return img


def vis_frame(frame, im_res, frame_count, debug, opt, format='coco'):
    """
    视频输出绘制
    :param frame: frame image
    :param im_res: im_res of predictions
    :param frame_count: 图片属于视频的第几帧
    :param debug: useless
    :param opt: options
    :param format: coco or mpii
    :return: rendered image
    """
    # 查看输入
    """
    global FLAG
    if FLAG:
        with open("./opt.txt", 'a') as fp:
            fp.write(str(opt))
        FLAG = False
    """
    # 获取图片属性
    img = frame.copy()
    height, width = img.shape[:2]
    # imgname = im_res['imgname']
    # 引入全局变量
    global count_frame_rate, angle, angle_info
    # 文本区域加一层灰色透明图
    background_left = np.zeros((150, 155, 3), dtype=np.uint8) + 125  # 灰色
    background_right = np.zeros((150, 160, 3), dtype=np.uint8) + 125  # 灰色
    roiLeft = img[0:130, 0:155]
    roiRight = img[0:130, width - 160:width]
    img[0:150, 0:155] = cv.addWeighted(background_left, 0.5, roiLeft, 0.5, 0)
    img[0:150, width - 160:width] = cv.addWeighted(background_right, 0.5, roiRight, 0.5, 0)
    del background_left, background_right, roiLeft, roiRight
    # 处理得到的数据
    if format != 'coco':  # 仅针对coco 17个关键点的注释格式
        pass

    count = len(im_res['result'])
    if count == 0:  # 没有检测到人
        return img

    elif count == 1:  # 只检测到一个人

        personAnno = {"bbox": [],
                      "keypoints": [],
                      "kpScores": [],
                      "id": 0,
                      "name": "",
                      "sex": ""}

        item = im_res['result'][0]

        try:
            # 处理bbox
            bbox = item['box']
            if len(bbox) != 4:
                text = "error. the len of bbox is not 4. \n"
                raise ValueError(text)
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # xmin,ymin,xmax,ymax
            bbox = list(map(int, bbox))
            personAnno['bbox'] = bbox
            # 处理keypoints
            keypoints = item['keypoints'].tolist()
            if len(keypoints) != 17:
                text = "error. the len of keypoints is not 17. \n"
                raise ValueError(text)
            for index in range(17):
                keypoints[index] = list(map(int, keypoints[index]))
            # 加一个点,总计十八个点
            x_neck = int((keypoints[5][0] + keypoints[6][0]) / 2)
            y_neck = int((keypoints[5][1] + keypoints[6][1]) / 2)
            keypoints.append([x_neck, y_neck])
            personAnno['keypoints'] = keypoints
            # 处理scores
            personAnno['kpScores'] = item['kp_score'].tolist()
            # 处理id
            personAnno['id'] = item['idx']
            # 姓名和性别
            personAnno['name'] = item['name']
            personAnno['sex'] = item['sex']
        except ValueError as e:
            print(e)

        del item

        Nose, LEye, REye, LEar, REar = personAnno['keypoints'][0:5:1]
        xmin = min(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
        xmax = max(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
        ymin = 0
        ymax = 0
        radius = (xmax - xmin) / 2

        for i in range(5):
            if personAnno['keypoints'][i][0] == xmin:
                ymin = int(abs(personAnno['keypoints'][i][1] - radius))
            if personAnno['keypoints'][i][0] == xmax:
                ymax = int(abs(personAnno['keypoints'][i][1] + radius))
        head_bbox = [xmin, ymin, xmax, ymax]  # xmin， ymin， xmax， ymax
        personAnno['head_bbox'] = head_bbox

        # Draw bboxes
        bbox = personAnno['bbox']
        if opt.showbox:
            # body_bbox
            single_bbox_draw(img, bbox)
            # head_bbox
            head_bbox = personAnno['head_bbox']
            cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)

        # Draw keypoints
        keypoints_draw(img, personAnno['keypoints'])
        # Draw limbs
        limbs_draw(img, personAnno['keypoints'])
        # Calc Keypoint
        if personAnno['id'] == 1:  # id = 1 的人画在左边
            loc = [0 + 5, 0]
            # 绘制另一个文本
            loc2 = [width - 160, 0]
            offset = 15
            for key in angle:
                text_bbox_draw(img, key + "--", [loc2[0] + 5, loc2[1] + offset])
                offset += 15
        else:  # id = 2的人画在右边
            loc = [width - 160, 0]
            loc1 = [0 + 5, 0]
            offset = 15
            for key in angle:
                text_bbox_draw(img, key + "--", [loc1[0] + 5, loc1[1] + offset])
                offset += 15

        kp_angle = {
            "left_shoulder": [personAnno['keypoints'][i] for i in (8, 6, 12)],
            "right_shoulder": [personAnno['keypoints'][i] for i in (7, 5, 11)],
            "left_hip": [personAnno['keypoints'][i] for i in (6, 12, 14)],
            "right_hip": [personAnno['keypoints'][i] for i in (5, 11, 13)],
            "left_knee": [personAnno['keypoints'][i] for i in (12, 14, 16)],
            "right_knee": [personAnno['keypoints'][i] for i in (11, 13, 15)]
        }

        # 实现每五帧更新一次文本
        if count_frame_rate == 0:
            # 更新angle
            angle['id'] = personAnno['id']
            angle['name'] = personAnno['name']
            angle['sex'] = personAnno['sex']

            for key, value in kp_angle.items():
                angle[key] = calc_kp_angle(value)

            loc_offset = 15
            for key, value in angle.items():
                if value != -1:
                    text_bbox_draw(img, key + ":" + str(value), [loc[0] + 5, loc[1] + loc_offset])
                else:
                    text_bbox_draw(img, key + ":" + "--", [loc[0] + 5, loc[1] + loc_offset])

                loc_offset += 15

        elif count_frame_rate == 5:
            count_frame_rate = 0
        else:
            count_frame_rate += 1

        # 读取json文件
        with open(json_file_path, 'r') as fp:
            info = json.load(fp)
            # print("读取output.json文件。")

        # 追加
        for item in info:
            if personAnno['id'] == item['idx']:
                for key, value in kp_angle.items():
                    item['angle'][key].append(calc_kp_angle(value))
                item['jump'].append(False)
                item['bbox'].append(personAnno['bbox'])
            else:
                for angle_name in angle_info:
                    item['angle'][angle_name].append(-1)
                item['jump'].append(False)
                bbox = [-1, -1, -1, -1]
                item['bbox'].append(bbox)

        with open(json_file_path, 'w') as fp:
            json.dump(info, fp)
            # print("覆盖写入output.json文件。")

        return img

    # count > 2 or count == 2
    # elif count > 2:
    #     # 获取其中最大的两个框覆盖im_res['result']
    #     res = {}
    #     for idx, item in enumerate(im_res['result']):
    #         area = item['box'][2] * item['box'][3]
    #         res[idx] = area
    #     res = sorted(res.items(), key=lambda _item: _item[1], reverse=True)
    #     # print(str(res))
    #     res1 = im_res['result'][res[0][0]]
    #     res2 = im_res['result'][res[1][0]]
    #     im_res['result'] = [res1, res2]
    #     del res, res1, res2
    elif count == 2:
        # 判断长度
        # assert len(im_res) == 2, "error, the len of im_res is not 2"
        personAnnos = [{"bbox": [],
                        "keypoints": [],
                        "kpScores": [],
                        "id": 0,
                        "name": "",
                        "sex": ""} for _ in range(2)]
        for i, item in enumerate(im_res['result']):
            try:
                # 处理bbox
                bbox = item['box']
                if len(bbox) != 4:
                    text = "error. the len of bbox is not 4. \n"
                    raise ValueError(text)
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # xmin,ymin,xmax,ymax
                bbox = list(map(int, bbox))
                personAnnos[i]['bbox'] = bbox
                # 处理keypoints
                keypoints = item['keypoints'].tolist()
                if len(keypoints) != 17:
                    text = "error. the len of keypoints is not 17. \n"
                    raise ValueError(text)
                for index in range(17):
                    keypoints[index] = list(map(int, keypoints[index]))
                # 加一个点,总计十八个点
                x_neck = int((keypoints[5][0] + keypoints[6][0]) / 2)
                y_neck = int((keypoints[5][1] + keypoints[6][1]) / 2)
                keypoints.append([x_neck, y_neck])
                personAnnos[i]['keypoints'] = keypoints
                # 处理scores
                personAnnos[i]['kpScores'] = item['kp_score'].tolist()
                # 处理id
                personAnnos[i]['id'] = item['idx']
                # 添加姓名和性别
                personAnnos[i]['name'] = item['name']
                personAnnos[i]['sex'] = item['sex']
            except ValueError as e:
                print(e)
        # for item in personAnnos:
        # for i, score in enumerate(item['kpScores']):
        # if score[0] < 0.1:  # 置信度
        #     item['keypoints'][i] = [-1, -1]
        # 清除变量
        del bbox, keypoints
        # 计算head_bbox
        for item in personAnnos:
            Nose, LEye, REye, LEar, REar = item['keypoints'][0:5:1]
            xmin = min(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            xmax = max(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            ymin = 0
            ymax = 0
            radius = (xmax - xmin) / 2
            for i in range(5):
                if item['keypoints'][i][0] == xmin:
                    ymin = int(abs(item['keypoints'][i][1] - radius))
                if item['keypoints'][i][0] == xmax:
                    ymax = int(abs(item['keypoints'][i][1] + radius))
            head_bbox = [xmin, ymin, xmax, ymax]  # xmin， ymin， xmax， ymax
            item['head_bbox'] = head_bbox

        # Draw bboxes
        if opt.showbox:
            # body_bbox
            bbox_draw(img, personAnnos[0]['bbox'], personAnnos[1]['bbox'])
            # head_bbox
            for item in personAnnos:
                head_bbox = item['head_bbox']
                cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)

        for index, item in enumerate(personAnnos):
            # 绘制标号
            """
            if opt.tracking:
                bbox = item['bbox']
                cv.putText(img, str(item['id']), (bbox[0] + 15, bbox[1] + 15), DEFAULT_FONT, 1, BLACK, 2)
            """
            # Draw keypoints
            keypoints_draw(img, item['keypoints'])
            # Draw limbs
            limbs_draw(img, item['keypoints'])
            # Calc Keypoint
            if item['id'] == 1:  # id = 1 的人画在左边
                loc = [0 + 5, 0]
            else:  # id = 2的人画在右边
                loc = [width - 160, 0]

            kp_angle = {
                "left_shoulder": [item['keypoints'][i] for i in (8, 6, 12)],
                "right_shoulder": [item['keypoints'][i] for i in (7, 5, 11)],
                "left_hip": [item['keypoints'][i] for i in (6, 12, 14)],
                "right_hip": [item['keypoints'][i] for i in (5, 11, 13)],
                "left_knee": [item['keypoints'][i] for i in (12, 14, 16)],
                "right_knee": [item['keypoints'][i] for i in (11, 13, 15)]
            }

            # 实现每五帧更新一次文本
            if count_frame_rate == 0:
                # 更新angle
                angle['id'] = item['id']
                angle['name'] = item['name']
                angle['sex'] = item['sex']

                for key, value in kp_angle.items():
                    angle[key] = calc_kp_angle(value)

                loc_offset = 15
                for key, value in angle.items():
                    if value != -1:
                        text_bbox_draw(img, key + ":" + str(value), [loc[0] + 5, loc[1] + loc_offset])
                    else:
                        text_bbox_draw(img, key + ":" + "--", [loc[0] + 5, loc[1] + loc_offset])
                    loc_offset += 15
            elif count_frame_rate == 5:
                count_frame_rate = 0
            else:
                count_frame_rate += 1

            # 读取json文件并追加
            with open(json_file_path, 'r') as fp:
                info = json.load(fp)
                # print("读取output.json文件。")

            # 追加
            for info_item in info:
                if item['id'] == info_item['idx']:
                    for key, value in kp_angle.items():
                        info_item['angle'][key].append(calc_kp_angle(value))
                    info_item['jump'].append(False)
                    info_item['bbox'].append(item['bbox'])

            with open(json_file_path, 'w') as fp:
                json.dump(info, fp)
                # print("覆盖写入output.json文件。")
        # 记录标号
        """
        if opt.tracking:
            with open("./id.txt", 'a') as fp:
                fp.write("id1:" + str(personAnnos[0]['id']) + "   " + "id2:" + str(personAnnos[1]['id']) + "\n")
        """
        """
        global FLAG
        if FLAG:
            with open("./personAnnos.txt", 'w') as fp:
                fp.write(str(personAnnos))
            FLAG = False
        """
        # print(f"视频的第{frame_count}帧")
        return img
