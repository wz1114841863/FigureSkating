import sys
import json
from copy import deepcopy
def judge_rotate():
    json_path = '/home/dell/wz/AlphaPose/examples/output/output_video.json'
    content = None
    with open(json_path) as f:
        content = json.load(f)
    if not content:
        print('json = None')
        sys.exit()

    # front = [[0,1], [1, 1], ...] 0,1代表面向方向
    front = []
    now_front = [0, 0]
    for i in range(len(content)):
        for j in range(len(content[i]['result'])):
            instance = content[i]['result'][j]
            now_kp = instance['keypoints']
            now_score = instance['kp_score']
            now_idx = int(instance['idx']) - 1
            if now_score[5][0] > 0.8 and now_score[6][0] > 0.8:
                now_front[now_idx] = 0 if now_kp[5][0] > now_kp[6][0] else 1
        front.append(deepcopy(now_front))


    front_0 = [i[0] for i in front]
    front_1 = [i[1] for i in front]
    rotate_frame_id = []
    for fro in [front_0, front_1]:
        front_dif = [abs(fro[i+1] - fro[i]) for i in range(len(front)-1)]
        dif_idx = [i for i in range(len(front_dif)) if front_dif[i] == 1]
        dif_dif = [dif_idx[i+1] - dif_idx[i] for i in range(len(dif_idx)-1)]

        # 连续2个以上间隔小于score的快速旋转
        score = 10
        rotate_dif_id = []
        now_rotate_dif_id = [-2, -2]
        for i in range(len(dif_dif)):
            if dif_dif[i] < score:
                if i > now_rotate_dif_id[1] + 1:
                    # 新序列
                    now_rotate_dif_id[0] = i
                    now_rotate_dif_id[1] = i
                else:
                    now_rotate_dif_id[1] = i
            else:
                if now_rotate_dif_id[1] - now_rotate_dif_id[0] > 1 \
                    and i == now_rotate_dif_id[1] + 1:
                    rotate_dif_id.append(deepcopy(now_rotate_dif_id))
        if now_rotate_dif_id[1] - now_rotate_dif_id[0] > 1 \
            and dif_dif[-1] < score:
            rotate_dif_id.append(deepcopy(now_rotate_dif_id))
        # 输出变换的帧数范围
        rotate_frame_id.append([[dif_idx[i[0]], dif_idx[i[1] + 1]] for i in rotate_dif_id])

    # rotate_frame_id取交集
    cornfield = [0 for i in range(len(front))]
    seeds = [*rotate_frame_id[0], *rotate_frame_id[1]]
    for seed in seeds:
        for i in range(seed[0], seed[1] + 1):
            cornfield[i] = 1

    x = list(range(len(cornfield)))
    # plt.plot(cornfield)
    # plt.show()

    json_path = '/home/dell/wz/AlphaPose/examples/output/output.json'
    content = None
    with open(json_path) as f:
        content = json.load(f)
    if not content:
        print('json = None')
        sys.exit()

    for i in range(len(cornfield)):
        rot = True if cornfield[i] == 1 else False
        for j in range(len(content)):
            content[j]['action']['rotate'].append(rot)

    with open(json_path, 'w') as f:
        json.dump(content, f)
