# Document function description
#   used to draw only one people in the image
from this import d
import cv2 as cv
from alphapose.utils import vis_video
import numpy as np
import redis



def draw_one_people(img, result):
    """according to result, dram info into img"""
    img_name = result['imgname']
    annos = result['result']
    angle = {
        "left_shoulder": 0.0,
        "right_shoulder": 0.0,
        "left_hip": 0.0,
        "right_hip": 0.0,
        "left_knee": 0.0,
        "right_knee": 0.0,
        "left_ankle": 0.0,
        "right_ankle": 0.0,
    }
    cnt = 0 
    for anno in annos:
        cnt = cnt + 1
        # draw bbox
        bbox = anno['box']
        assert len(bbox) == 4, f"the length of bbox not equal to 4"
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        vis_video.single_bbox_draw(img, bbox)
        # draw keypoints
        keypoints = np.array(anno['keypoints'], dtype=int)
        assert keypoints.shape[0] == 17, f"the shape of keyppoints error, {keypoints.shape}"
        vis_video.keypoints_draw(img, keypoints)
        # draw limb
        vis_video.limbs_draw(img, keypoints)
        # calc angle
        kp_angle = {
            "left_shoulder": [keypoints[i] for i in (7, 5, 11)],
            "right_shoulder": [keypoints[i] for i in (8, 6, 12)],
            "left_hip": [keypoints[i] for i in (5, 11, 13)],
            "right_hip": [keypoints[i] for i in (6, 12, 14)],
            "left_knee": [keypoints[i] for i in (11, 13, 15)],
            "right_knee": [keypoints[i] for i in (12, 14, 16)],
            "left_ankle": [keypoints[13], keypoints[15], vis_video.calc_ankle(keypoints[15])],
            "right_ankle":[keypoints[14], keypoints[16], vis_video.calc_ankle(keypoints[16])],
        }
        for key, value in kp_angle.items():
            angle[key] = vis_video.calc_kp_angle(value)
        if cnt == 24:
            #wirte_into_queue(redis_conn, 'pose_queue', str(angle))
            cnt = 0
        
    return img, angle
    


if __name__ == '__main__':
    pass