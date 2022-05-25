import math
import cv2 as cv
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont

from .constantPath import *

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
DEFAULT_FONT = cv.FONT_HERSHEY_TRIPLEX
# annotations：the number of keypoints is 17
limbPairs = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),  # Body
             (11, 13), (12, 14), (13, 15), (14, 16)]  # Foot
# 躯干颜色
limbPairsColor = [CYAN for _ in range(15)]
# 关键点名称
keyPointsNames = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
# 标志位
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
    "right_knee": 0.0,
    "left_ankle": 0.0,
    "right_ankle": 0.0,
}
# 需要标注角度的关节点名称
angle_info = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", 
              "left_knee", "right_knee", "left_ankle", "right_ankle"]


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


def single_bbox_draw(_img, _bbox, _color=BLUE_1):
    """
    绘制一个人的bbo以及角线
    :param _img  原始图片
    :param _bbox  bbox  [xmin, ymin, xmax, ymax]
    :param _color 默认颜色
    """
    cv.rectangle(_img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), _color, 2)
    # 画直线来填充边框
    _color_1 = CYAN
    _color_2 = WHITE
    thickness_1 = 5
    thickness_2 = 3

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


def calc_kp_center(keypoints):
    """
    计算中心点坐标
    :param keypoints:
    :return kp_center: 中心点坐标
    """
    kp_center = [0, 0]
    length = 0
    for item in keypoints:
        if item[0] != -1:
            kp_center[0] = kp_center[0] + item[0]
            kp_center[1] = kp_center[1] + item[1]
            length += 1
    kp_center = [int(kp_center[0] / length), int(kp_center[1] / length)]
    return kp_center


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
    angle_A  += 0.001
    angle_A = round(angle_A, 2)
    return angle_A


def limbs_draw(img, keypoints):
    """
    绘制关键点之间的连线。
    :param img: 原始图片
    :param keypoints: 绘制所有存在的关节对。[[x1, y1], [x2, y2], ...]
    :return: None
    """
    for i, pair in enumerate(limbPairs[4:]):  # 不绘制脸上的线
        if keypoints[pair[0]][0] != -1 and keypoints[pair[1]][0] != -1:
            if i + 4 not in (4, 6, 8):
                # cv.line(img, tuple(keypoints[pair[0]]), tuple(keypoints[pair[1]]), limbPairsColor[i + 4], 2)
                cv.line(img, tuple(keypoints[pair[0]]), tuple(keypoints[pair[1]]), WHITE, 2)
            else:
                cv.line(img, tuple(keypoints[pair[0]]), tuple(keypoints[pair[1]]), WHITE, 2)


def text_draw(img, keypoint, text):
    """
    绘制文本，描述关节点的角度。
    :param img: 原始图片
    :param keypoint: 要绘制的关键节点
    :param text: 要绘制的文本
    :return: 绘制后的图片
    """
    # 检验关节点的有效性
    if keypoint[0] == -1:
        return img
    cv.line(img, tuple(keypoint), (keypoint[0] - 50, keypoint[1] - 25), WHITE, 2)
    cv.line(img, (keypoint[0] - 50, keypoint[1] - 25), (keypoint[0] - 100, keypoint[1] - 25), WHITE, 2)
    # cv.putText(img, text, (keypoint[0] - 100, keypoint[1] - 25), DEFAULT_FONT, 0.5, WHITE, 2, cv.LINE_AA)
    location = (keypoint[0] - 100, keypoint[1] - 45)
    img = text_bbox_draw(img, text, location, textColor=(255, 0, 0))
    return img


def text_draw_loc(img, keypoint, offset, text):
    """
    绘制文本，描述关节点的角度。
    :param img: 原始图片
    :param keypoint: 要绘制的关键节点
    :param offset: 偏移方向
    :param text: 要绘制的文本
    :return: 绘制后的图片
    """
    # 检验关节点的有效性
    if keypoint[0] == -1:
        return img
    # 根据图片比例线长
    height, width = img.shape[:2]
    # line_height = int(height * 0.2)
    # line_width = int(width * 0.2)
    line_height = 100
    line_width = 150
    if offset == 0:
        cv.line(img, tuple(keypoint), (keypoint[0] - line_width, keypoint[1] - line_height), WHITE, 2)
        # cv.putText(img, text, (keypoint[0] - 100, keypoint[1] - 25), DEFAULT_FONT, 0.5, WHITE, 2, cv.LINE_AA)
        location = (keypoint[0] - line_width - 50, keypoint[1] - line_height)
        img = text_bbox_draw(img, text, location, textColor=(255, 0, 0))
    else:
        cv.line(img, tuple(keypoint), (keypoint[0] + line_width, keypoint[1] - line_height), WHITE, 2)
        # cv.putText(img, text, (keypoint[0] - 100, keypoint[1] - 25), DEFAULT_FONT, 0.5, WHITE, 2, cv.LINE_AA)
        location = (keypoint[0] + line_width, keypoint[1] - line_height)
        img = text_bbox_draw(img, text, location, textColor=(255, 0, 0))
    return img


def keypoints_draw(img, keypoints, color=ORANGE):
    """
    绘制关键点
    :param img: 原始图片
    :param keypoints: 人体关键点 [[x1, y1], [x2, y2], ...]
    :param color: 关键点的颜色
    :return: None
    """
    # 首先绘制所有可能的关键点
    for i, kp in enumerate(keypoints[5:]):  # 不绘制脸部的点
        if kp[0] != -1:  # 关键点坐标存在
            if i + 4 in (5, 6, 11, 12, 13, 14):
                cv.circle(img, (int(kp[0]), int(kp[1])), 4, WHITE, -1)
            else:
                cv.circle(img, (int(kp[0]), int(kp[1])), 2, WHITE, -1)


def gray_mask_draw(img):
    """
    绘制灰色蒙版区域
    :param img: 输入的原始图片
    :return: None
    """
    height, width = img.shape[:2]
    left_mask_width = int(0.16 * width)
    left_mask_height = int(0.27 * height)
    right_mask_width = left_mask_width + 10
    right_mask_height = left_mask_height

    background_left = np.zeros((left_mask_height, left_mask_width, 3), dtype=np.uint8) + 125  # 灰色
    background_right = np.zeros((right_mask_height, right_mask_width, 3), dtype=np.uint8) + 125  # 灰色
    roiLeft = img[0:left_mask_height, 0:left_mask_width]
    roiRight = img[0:right_mask_height, width - right_mask_width:width]
    img[0:left_mask_height, 0:left_mask_width] = cv.addWeighted(background_left, 0.5, roiLeft, 0.5, 0)
    img[0:right_mask_height, width - right_mask_width:width] = cv.addWeighted(background_right, 0.5, roiRight, 0.5, 0)


def text_bbox_draw(img, text, location, textColor=(255, 255, 255)):
    """
    绘制人物信息
    :param img: 原始图片
    :param text: 要绘制的文本
    :param location: 要绘制文本的位置 [x, y]
    :param textColor: 字体颜色
    :return img: numpy
    """
    height, width = img.shape[:2]
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # textSize = int(min(width * 0.015, height * 0.027))
    textSize = 30
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("/home/dell/wz/AlphaPose/alphapose/utils/Fonts/Deng.ttf", textSize,
                                  encoding="utf-8")
    draw.text(location, text, textColor, font=fontText)

    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def calc_ankle(angle1):
    """
    计算。
    :param angle1: 踝关节位置坐标 [x, y]。
    :return angle: 踝关节位置坐标平移后的坐标 [x, y]。
    """
    if angle1[0] == -1:
        return [-1, -1]
    else:
        return [angle1[0] + 10, angle1[1]]


def empty_txt_draw(img):
    """
    在img的两边绘制空的文本。
    :param img: 原始图片
    :return: img
    """
    global angle
    height, width = img.shape[:2]
    # 偏移量
    right_offset = int(width * 0.16)
    # 绘制两个文本
    loc1 = [0 + 5, 0]
    loc2 = [width - right_offset, 0]
    offset = 10
    for key in angle:
        for item in (loc1, loc2):
            img = text_bbox_draw(img, key + "--", [item[0] + 5, item[1] + offset])
        offset += int(0.027 * height)
    return img


def vis_frame_empty(frame):
    """
    没有预测到bbox的帧处理
    :param frame: 输入的原始图片
    :return: img
    """
    img = frame.copy()
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print("读取output_source.json文件。")

    global angle_info

    # 不用区分人物id，直接写入json
    for item in info:
        for angle_name in angle_info:
            item['angle'][angle_name].append(-1)
        bbox = [-1, -1, -1, -1]  # xmin, ymin, xmax, ymax
        item['bbox'].append(bbox)

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")

    return img


def  vis_frame(frame, im_res):
    """
    视频输出绘制
    :param frame: 原始图片
    :param im_res: 注释
    :return img: 绘制后的图像
    """
    # 图片属性
    img = frame.copy()
    height, width = img.shape[:2]
    # 引入全局变量
    global count_frame_rate, angle, angle_info
    # 判断识别到的人的个数
    count = len(im_res['result'])
    assert count in (1, 2), "the count is not one or two"
    personAnnos = [{"id": i + 1,
                    "name": "",
                    "sex": "",
                    "bbox": [],
                    "keypoints": [],
                    "kpScores": []} for i in range(2)]
    # 添加一个flag来标志对应人的信息是否有效
    person_flag = [False, False]
    # 不管识别到几个人，统一处理info
    for i, item in enumerate(im_res['result']):
        try:
            idx = item['idx'] - 1
            # 处理bbox
            bbox = item['box']
            if len(bbox) != 4:
                text = "error. the len of bbox is not 4. \n"
                raise ValueError(text)
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # xmin,ymin,xmax,ymax
            bbox = list(map(int, bbox))
            personAnnos[idx]['bbox'] = bbox
            # 处理keypoints
            keypoints = item['keypoints']
            if len(keypoints) != 17:
                text = "error. the len of keypoints is not 17. \n"
                raise ValueError(text)
            for index in range(17):
                keypoints[index] = list(map(int, keypoints[index]))
            # 加一个点,总计十八个点
            x_neck = int((keypoints[5][0] + keypoints[6][0]) / 2)
            y_neck = int((keypoints[5][1] + keypoints[6][1]) / 2)
            keypoints.append([x_neck, y_neck])
            personAnnos[idx]['keypoints'] = keypoints
            # 处理scores
            personAnnos[idx]['kpScores'] = item['kp_score']
            # 处理id
            # personAnnos[idx]['id'] = item['idx']
            assert personAnnos[idx]['id'] == item['idx'], "personAnnos'id is not equal the item['idx']"
            # 添加姓名和性别
            personAnnos[idx]['name'] = item['name']
            personAnnos[idx]['sex'] = item['sex']
            # 记录标志位
            person_flag[idx] = True
            # 添加head_bbox
            Nose, LEye, REye, LEar, REar = personAnnos[idx]['keypoints'][0:5:1]
            xmin = min(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            xmax = max(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            ymin = 0
            ymax = 0
            radius = (xmax - xmin) / 2
            for j in range(5):
                if personAnnos[idx]['keypoints'][j][0] == xmin:
                    ymin = int(abs(personAnnos[idx]['keypoints'][j][1] - radius))
                if personAnnos[idx]['keypoints'][j][0] == xmax:
                    ymax = int(abs(personAnnos[idx]['keypoints'][j][1] + radius))
            head_bbox = [xmin, ymin, xmax, ymax]  # xmin， ymin， xmax， ymax
            personAnnos[idx]['head_bbox'] = head_bbox

        except ValueError as e:
            print(e)

    del bbox, head_bbox, keypoints, idx, item
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print(f"info:{info}")

    # 绘制关键点和躯干
    for i, item in enumerate(personAnnos):
        # 偏移量
        right_offset = int(width * 0.16)
        # 绘制文本位置
        if item['id'] == 1:  # id = 1 的人画在左边
            loc = [0 + 5, 0]
        elif item['id'] == 2:  # id = 2的人画在右边
            loc = [width - right_offset, 0]
        else:
            raise ValueError("the value of 'item['id']' Beyond expectation。")

        if person_flag[i]:  # 当前人物数据有效

            # Draw keypoints
            keypoints_draw(img, item['keypoints'])
            # Draw limbs
            limbs_draw(img, item['keypoints'])
            # 需要计算角度的关键点
            kp_angle = {
                "left_shoulder": [item['keypoints'][i] for i in (7, 5, 11)],
                "right_shoulder": [item['keypoints'][i] for i in (8, 6, 12)],
                "left_hip": [item['keypoints'][i] for i in (5, 11, 13)],
                "right_hip": [item['keypoints'][i] for i in (6, 12, 14)],
                "left_knee": [item['keypoints'][i] for i in (11, 13, 15)],
                "right_knee": [item['keypoints'][i] for i in (12, 14, 16)],
                "left_ankle": [item['keypoints'][13], item['keypoints'][15], calc_ankle(item['keypoints'][15])],
                "right_ankle":[item['keypoints'][14], item['keypoints'][16], calc_ankle(item['keypoints'][16])],
            }
            # 追加
            for info_item in info:
                if item['id'] == info_item['idx']:
                    for key, value in kp_angle.items():
                        info_item['angle'][key].append(calc_kp_angle(value))
                    info_item['bbox'].append(item['bbox'])
                else:
                    continue
        else:  # 当前人物注释无效
            # 追加
            for info_item in info:
                if item['id'] == info_item['idx']:
                    for angle_name in angle_info:
                        info_item['angle'][angle_name].append(-1)
                    bbox = [-1, -1, -1, -1]
                    info_item['bbox'].append(bbox)
                else:
                    continue
    del item

    # Draw bboxs
    if count == 1:  # 只识别到一个人的bbox
        #
        for i, item in enumerate(personAnnos):
            if person_flag[i]:  # 当前人物数据有效
                bbox = item['bbox']
                single_bbox_draw(img, bbox)
                head_bbox = item['head_bbox']
                cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)
                # 绘制指示文本
                bbox_center = int((bbox[0] + bbox[3]) / 2)
                # 计算中心关键点
                kp_center = calc_kp_center(item['keypoints'])
                text = str(calc_kp_angle([item['keypoints'][i] for i in (12, 14, 16)]))
                offset = 0
                img = text_draw_loc(img, kp_center, offset, item['name'])
                img = text_draw_loc(img, item['keypoints'][14], offset, text)

            else:
                continue
    elif count == 2:
        # body_bbox
        bbox_draw(img, personAnnos[0]['bbox'], personAnnos[1]['bbox'])
        # 计算两个人的bbox中心点
        bbox_0_center = int((personAnnos[0]['bbox'][0] + personAnnos[0]['bbox'][3]) / 2)
        bbox_1_center = int((personAnnos[1]['bbox'][0] + personAnnos[0]['bbox'][3]) / 2)
        personAnnos[0]['bbox_center'] = bbox_0_center
        personAnnos[1]['bbox_center'] = bbox_1_center
        bbox_left = min(bbox_0_center, bbox_1_center)
        # bbox_right = max(bbox_0_center, bbox_1_center)
        # head_bbox
        for item in personAnnos:
            head_bbox = item['head_bbox']
            cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)
            # 绘制指示文本
            # 计算中心关键点
            kp_center = calc_kp_center(item['keypoints'])
            text = str(calc_kp_angle([item['keypoints'][i] for i in (12, 14, 16)]))
            if item['bbox_center'] == bbox_left:
                offset = 0
            else:
                offset = 1
            img = text_draw_loc(img, kp_center, offset, item['name'])
            img = text_draw_loc(img, item['keypoints'][14], offset, text)

            del item['bbox_center']
    else:
        raise ValueError("the value of 'count' Beyond expectation。")

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")
    return img


def vis_frame_empty_1(frame):
    """
    没有预测到bbox的帧，绘制文本框
    :param frame: 输入的原始图片
    :return: img
    """
    img = frame.copy()
    # 添加灰色蒙版
    gray_mask_draw(img)
    # 绘制空白文本
    img = empty_txt_draw(img)
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print("读取output_source.json文件。")

    global angle_info

    # 不用区分人物id，直接写入json
    for item in info:
        for angle_name in angle_info:
            item['angle'][angle_name].append(-1)
        item['jump'].append(False)
        bbox = [-1, -1, -1, -1]  # xmin, ymin, xmax, ymax
        item['bbox'].append(bbox)

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")

    return img


def vis_frame_1(frame, im_res):
    """
    视频输出绘制
    :param frame: 原始图片
    :param im_res: 注释
    :return img: 绘制后的图像
    """
    # 图片属性
    img = frame.copy()
    height, width = img.shape[:2]
    # 添加灰色蒙版
    gray_mask_draw(img)
    #
    # cv.putText(img, im_res['imgname'] + str(im_res['debug']), (5, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # 引入全局变量
    global count_frame_rate, angle, angle_info
    # 判断识别到的人的个数
    count = len(im_res['result'])
    assert count in (1, 2), "the count is not one or two"
    personAnnos = [{"id": i + 1,
                    "name": "",
                    "sex": "",
                    "bbox": [],
                    "keypoints": [],
                    "kpScores": []} for i in range(2)]
    # 添加一个flag来标志对应人的信息是否有效
    person_flag = [False, False]
    # 不管识别到几个人，统一处理info
    for i, item in enumerate(im_res['result']):
        try:
            idx = item['idx'] - 1
            # 处理bbox
            bbox = item['box']
            if len(bbox) != 4:
                text = "error. the len of bbox is not 4. \n"
                raise ValueError(text)
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # xmin,ymin,xmax,ymax
            bbox = list(map(int, bbox))
            personAnnos[idx]['bbox'] = bbox
            # 处理keypoints
            keypoints = item['keypoints']
            if len(keypoints) != 17:
                text = "error. the len of keypoints is not 17. \n"
                raise ValueError(text)
            for index in range(17):
                keypoints[index] = list(map(int, keypoints[index]))
            # 加一个点,总计十八个点
            x_neck = int((keypoints[5][0] + keypoints[6][0]) / 2)
            y_neck = int((keypoints[5][1] + keypoints[6][1]) / 2)
            keypoints.append([x_neck, y_neck])
            personAnnos[idx]['keypoints'] = keypoints
            # 处理scores
            personAnnos[idx]['kpScores'] = item['kp_score']
            # 处理id
            # personAnnos[idx]['id'] = item['idx']
            assert personAnnos[idx]['id'] == item['idx'], "personAnnos'id is not equal the item['idx']"
            # 添加姓名和性别
            personAnnos[idx]['name'] = item['name']
            personAnnos[idx]['sex'] = item['sex']
            # 记录标志位
            person_flag[idx] = True
            # 添加head_bbox
            Nose, LEye, REye, LEar, REar = personAnnos[idx]['keypoints'][0:5:1]
            xmin = min(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            xmax = max(Nose[0], LEye[0], REye[0], LEar[0], REar[0])
            ymin = 0
            ymax = 0
            radius = (xmax - xmin) / 2
            for j in range(5):
                if personAnnos[idx]['keypoints'][j][0] == xmin:
                    ymin = int(abs(personAnnos[idx]['keypoints'][j][1] - radius))
                if personAnnos[idx]['keypoints'][j][0] == xmax:
                    ymax = int(abs(personAnnos[idx]['keypoints'][j][1] + radius))
            head_bbox = [xmin, ymin, xmax, ymax]  # xmin， ymin， xmax， ymax
            personAnnos[idx]['head_bbox'] = head_bbox

        except ValueError as e:
            print(e)

    del bbox, head_bbox, keypoints, idx, item
    # 读取json文件并追加
    with open(json_source_path, 'r') as fp:
        info = json.load(fp)
        # print(f"info:{info}")
    # print("读取output_source.json文件。")

    # 绘制关键点和躯干
    for i, item in enumerate(personAnnos):
        # 偏移量
        right_offset = int(width * 0.16)
        # 绘制文本位置
        if item['id'] == 1:  # id = 1 的人画在左边
            loc = [0 + 5, 0]
        elif item['id'] == 2:  # id = 2的人画在右边
            loc = [width - right_offset, 0]
        else:
            raise ValueError("the value of 'item['id']' Beyond expectation。")

        if person_flag[i]:  # 当前人物数据有效

            # Draw keypoints
            keypoints_draw(img, item['keypoints'])
            # Draw limbs
            limbs_draw(img, item['keypoints'])
            # 需要计算角度的关键点
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

            elif count_frame_rate == 5:  # 每五帧
                count_frame_rate = 0
            else:
                count_frame_rate += 1
            # 绘制文本
            loc_offset = 10
            for key, value in angle.items():
                if value != -1:  # 非角度值或角度返回值是有效值
                    img = text_bbox_draw(img, key + ":" + str(value), [loc[0] + 5, loc[1] + loc_offset])
                else:
                    img = text_bbox_draw(img, key + ":" + "--", [loc[0] + 5, loc[1] + loc_offset])

                loc_offset += int(0.027 * height)
            # 追加
            for info_item in info:
                if item['id'] == info_item['idx']:
                    for key, value in kp_angle.items():
                        info_item['angle'][key].append(calc_kp_angle(value))
                    info_item['jump'].append(False)
                    info_item['bbox'].append(item['bbox'])
                else:
                    continue
        else:  # 当前人物注释无效
            # 追加
            for info_item in info:
                if item['id'] == info_item['idx']:
                    for angle_name in angle_info:
                        info_item['angle'][angle_name].append(-1)
                    info_item['jump'].append(False)
                    bbox = [-1, -1, -1, -1]
                    info_item['bbox'].append(bbox)
                else:
                    continue
            # 文本绘制
            offset = 10
            for key in angle:
                img = text_bbox_draw(img, key + "--", [loc[0] + 5, loc[1] + offset])
                offset += int(0.027 * height)
    del item

    # Draw bboxs
    if count == 1:  # 只识别到一个人的bbox
        #
        for i, item in enumerate(personAnnos):
            if person_flag[i]:  # 当前人物数据有效
                bbox = item['bbox']
                single_bbox_draw(img, bbox)
                head_bbox = item['head_bbox']
                cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)
                # 绘制指示文本
                bbox_center = int((bbox[0] + bbox[3]) / 2)
                # 计算中心关键点
                kp_center = calc_kp_center(item['keypoints'])
                text = str(calc_kp_angle([item['keypoints'][i] for i in (12, 14, 16)]))
                offset = 0
                img = text_draw_loc(img, kp_center, offset, item['name'])
                img = text_draw_loc(img, item['keypoints'][14], offset, text)

            else:
                continue
    elif count == 2:
        # body_bbox
        bbox_draw(img, personAnnos[0]['bbox'], personAnnos[1]['bbox'])
        # 计算两个人的bbox中心点
        bbox_0_center = int((personAnnos[0]['bbox'][0] + personAnnos[0]['bbox'][3]) / 2)
        bbox_1_center = int((personAnnos[1]['bbox'][0] + personAnnos[0]['bbox'][3]) / 2)
        personAnnos[0]['bbox_center'] = bbox_0_center
        personAnnos[1]['bbox_center'] = bbox_1_center
        bbox_left = min(bbox_0_center, bbox_1_center)
        # bbox_right = max(bbox_0_center, bbox_1_center)
        # head_bbox
        for item in personAnnos:
            head_bbox = item['head_bbox']
            cv.rectangle(img, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), BLUE_2, 1)
            # 绘制指示文本
            # 计算中心关键点
            kp_center = calc_kp_center(item['keypoints'])
            text = str(calc_kp_angle([item['keypoints'][i] for i in (12, 14, 16)]))
            if item['bbox_center'] == bbox_left:
                offset = 0
            else:
                offset = 1
            img = text_draw_loc(img, kp_center, offset, item['name'])
            img = text_draw_loc(img, item['keypoints'][14], offset, text)

            del item['bbox_center']
    else:
        raise ValueError("the value of 'count' Beyond expectation。")

    with open(json_source_path, 'w') as fp:
        json.dump(info, fp)
        # print("覆盖写入output_source.json文件。")
    return img


if __name__ == '__main__':
    pass
