# Document function description
#   from alphapose-results.json file read corrdinate of keypoints
#   generate heatmap from keypoints and draw on the video
import json
import os
import re
import cv2 as cv
import numpy as np

json_path = "/home/dell/wz/AlphaPose/examples/output/output_video.json"
video_path = "/home/dell/wz/AlphaPose/examples/video/01.mp4"
video_outout_path = "/home/dell/wz/AlphaPose/examples/output/heatmap.mp4"


def get_heat_val(sigma, x, y, x0, y0):
    # Gaussian distribution
    val = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return val


def generate_heatmap(img, annos):
    """input keypoint annos, create a heatmap"""
    # img: [H, W, C]
    # keypoint: 17
    # heatmap: [17, H, W]
    H, W, _ = img.shape
    sigma = 4
    heatamp = np.zeros((17, H, W), dtype=np.float32)
    for anno in annos['result']:
        kps = anno['keypoints']
        for i in range(5, 17):
            x0, y0 = kps[i][0], kps[i][1]
            # get the valid area of keypoint heatmap
            ul = int(np.floor(x0 - 3 * sigma - 1)), int(np.floor(y0 - 3 * sigma - 1))
            br = int(np.ceil(x0 + 3 * sigma + 2)), int(np.ceil(y0 + 3 * sigma + 2))

            cc, dd = max(0, ul[0]), min(br[0], W)
            aa, bb = max(0, ul[1]), min(br[1], H)
            joint_gaus = np.zeros((bb - aa, dd - cc))
            for y in range(aa, bb):
                for x in range(cc, dd):
                    joint_gaus[y - aa, x - cc] = get_heat_val(sigma, x, y, x0, y0)
                    
            heatamp[i, aa:bb, cc:dd] = np.maximum(heatamp[i, aa:bb, cc:dd], joint_gaus)
                
    return heatamp


def draw_heatmap(img, heatmap):
    """draw heatmap into img"""
    # img [H, W, C]
    # heatmap: [17, H, W]
    heatmap = heatmap * 255
    heatmap = np.array(heatmap, np.uint8)  # (17, 683, 3)
    res = 0
    for hm in heatmap:
        res += hm
    heatmap = cv.applyColorMap(res, cv.COLORMAP_JET)
    # heatmap = cv.applyColorMap(res, cv.COLORMAP_HOT)
    result = img * 0.5 + heatmap * 0.5  # (H, W, 3)
    result = np.array(result, np.uint8)
    return result


if __name__ == '__main__':
    # load annos
    with open(json_path, 'r') as fp:
        annos = json.load(fp)
    capture = cv.VideoCapture(video_path)
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
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_result = cv.VideoWriter(video_outout_path, fourcc, frame_fps, (frame_width, frame_height), True)
    len_annos = len(annos)
    print(f"len(annos):{len_annos}")
    # Frame by frame drawing
    for idx in range(len_annos):
        # get current frame
        ret, img = capture.read()
        if not ret:
            print("get frame failede.")
            break
        heatmap = generate_heatmap(img, annos[idx])
        result = draw_heatmap(img, heatmap)
        # cv.imwrite('./test.jpg', result)
        # save frame
        video_result.write(result)

    # release video handle
    capture.release()
    video_result.release()
    # use ffmpeg to modoify video format 
    # ffmp = "ffmpeg -i " + video_outout_path + " -b:v 5M -vcodec libx264 -y -v quiet " + video_outout_path
    # os.system(ffmp)
    print("finish.")
