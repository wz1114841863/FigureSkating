import os
import sys
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose.utils.transforms import get_func_heatmap_to_coord

import json

sys.path.append("/home/dell/wz/AlphaPose/server")
from server.req import return_name

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

        # kalman filter
        # self.kalmans = [[],[]]
        # for i in range(2):
        #     for j in range(19):
        #         kalman = cv2.KalmanFilter(4,2)
        #         #设置测量矩阵
        #         kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        #         #设置转移矩阵
        #         kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        #         #设置过程噪声协方差矩阵
        #         kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.03
        #         self.kalmans[i].append(kalman)

    def kal(self, coor, kalman):
        current_measurement = np.array([[np.float32(coor[0])], [np.float32(coor[1])]])
        kalman.correct(current_measurement)
        current_prediction = kalman.predict()
        return [current_prediction[0].item(), current_prediction[1].item()]

    # def kalman_do(self, pre_res, cur_res):
    #     if len(cur_res['result']) == 2 and len(pre_res['result']) == 1 :

    # if len(cur_res['result']) != 2 or len(pre_res['result']) != 2 :
    #     return cur_res

    # # id reallocate
    # # if {cur_res['result'][0]['idx'], cur_res['result'][1]['idx']} == {{pre_res['result'][0]['idx'], pre_res['result'][1]['idx']}}:
    # cur_res['result'][0]['idx'] = cur_res['result'][0]['idx']%2
    # cur_res['result'][1]['idx'] = 1 - cur_res['result'][0]['idx']

    # for per in range(2):
    #     # box:17-18
    #     lu = [cur_res['result'][per]["box"][0], cur_res['result'][per]["box"][1]]
    #     rb = [cur_res['result'][per]["box"][0] + cur_res['result'][per]["box"][2],
    #           cur_res['result'][per]["box"][0] + cur_res['result'][per]["box"][3]]
    #     lu = self.kal(lu, self.kalmans[per][17])
    #     rb = self.kal(rb, self.kalmans[per][18])
    #     cur_res['result'][per]["box"][0] = lu[0]
    #     cur_res['result'][per]["box"][1] = lu[1]
    #     cur_res['result'][per]["box"][2] = rb[0] - lu[0]
    #     cur_res['result'][per]["box"][3] = rb[1] - lu[1]

    #     # pose:0-16
    #     for idx in range(17):
    #         if cur_res['result'][per]['kp_score'][idx] < 0.5:
    #             cur_res['result'][per]['keypoints'][idx] = pre_res['result'][per]['keypoints'][idx]
    #         coor = [cur_res['result'][per]['keypoints'][idx][0].item(), cur_res['result'][per]['keypoints'][idx][1].item()]
    #         coor = self.kal(coor, self.kalmans[per][idx])
    #         cur_res['result'][per]['keypoints'][idx] = torch.tensor(coor)

    # return cur_res

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            # p = mp.Process(target=target, args=())
            p = Thread(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
            #     stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            # assert stream.isOpened(), 'Cannot open video for writing'

        # wz
        # 记录帧数
        frame_count = 0
        # 创建json文件
        json_source_path = "//home/dell/wz/AlphaPose/examples/output/output_source.json"
        info = [{"idx": i + 1,
                 "name": "",
                 "sex": "",
                 "angle": {
                     "left_shoulder": [],
                     "right_shoulder": [],
                     "left_hip": [],
                     "right_hip": [],
                     "left_knee": [],
                     "right_knee": [],
                     "left_ankle": [],
                     "right_ankle": [],
                 },
                 "action":{
                    "jump": [],
                    "lift": [],
                    "rotate": [],
                 },
                 "bbox": [],
                 } for i in range(2)]

        # 写入名字和性别
        # data = return_name()
        # data_0 = data['0']
        # data_1 = data['1']
        # info[0]['name'] = data_0['name']
        # info[0]['sex'] = data_0['sex']
        # info[1]['name'] = data_1['name']
        # info[1]['sex'] = data_1['sex']

        with open(json_source_path, 'w') as fp:
            json.dump(info, fp)
            print("创建output_source.json文件。")

        del info
        # 存储所有的注释信息
        info = []
        json_video_path = "/home/dell/wz/AlphaPose/examples/output/output_video.json"
        # keep looping infinitelyd
        while True:
            
            frame_count += 1
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, debug) = self.wait_and_get(
                self.result_queue)
            if orig_img is None:
                frame_count -= 1
                # if the thread indicator variable is set (img is None), stop the thread
                # if self.save_video:
                #     stream.release()
                # write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                # 添加json处理函数
                # from .jsonDataProc import json_data_process
                # json_data_process()
                with open(json_video_path, 'w') as fp:
                    json.dump(info, fp)

                # print("Results have been written to json.")
                print(f"视频一共有{frame_count}帧")
                return

            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                result = {
                    'imgname': im_name,
                    'result': []
                }
                info.append(result)

                # if self.opt.save_img or self.save_video or self.opt.vis:
                #     # wz---begin
                #     from alphapose.utils.vis import vis_frame_empty
                #     img = vis_frame_empty(orig_img, frame_count)
                #     self.write_image(img, im_name, stream=stream if self.save_video else None)

            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                # pred = hm_data.cpu().data.numpy()

                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0, 136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0, 26)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox,
                                                                   hm_shape=hm_size,
                                                                   norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints': preds_img[k],
                            'kp_score': preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(
                                preds_scores[k]),
                            'idx': ids[k],
                            'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }

                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                # kal
                # if final_result:
                #     result = self.kalman_do(final_result[-1], result)
                # else:
                #     result['result'][0]['idx'] = result['result'][0]['idx']%2
                #     result['result'][1]['idx'] = 1 - result['result'][0]['idx']

                final_result.append(result)

                # if self.opt.save_img or self.save_video or self.opt.vis:
                #     if hm_data.size()[1] == 49:
                #         from alphapose.utils.vis import vis_frame_dense as vis_frame
                #     elif self.opt.vis_fast:
                #         from alphapose.utils.vis import vis_frame_fast as vis_frame
                #     else:
                #         from alphapose.utils.vis import vis_frame

                # 结果中添加name 和 id
                length = len(result['result'])
                if length == 1:
                    # if result['result'][0]['idx'] == 1:
                    #     result['result'][0]['name'] = data_0['name']
                    #     result['result'][0]['sex'] = data_0['sex']
                    # elif result['result'][0]['idx'] == 2:
                    #     result['result'][0]['name'] = data_1['name']
                    #     result['result'][0]['sex'] = data_1['sex']
                    # else:
                    #     print("error. the result id is not 1 or 2. ")
                    result['result'][0]['name'] = ""
                    result['result'][0]['sex'] = ""
                elif length == 2:
                    # if result['result'][0]['idx'] == 1:
                    #     result['result'][0]['name'] = data_0['name']
                    #     result['result'][0]['sex'] = data_0['sex']
                    #     result['result'][1]['name'] = data_1['name']
                    #     result['result'][1]['sex'] = data_1['sex']
                    # elif result['result'][0]['idx'] == 2:
                    #     result['result'][0]['name'] = data_1['name']
                    #     result['result'][0]['sex'] = data_1['sex']
                    #     result['result'][1]['name'] = data_0['name']
                    #     result['result'][1]['sex'] = data_0['sex']
                    result['result'][0]['name'] = ""
                    result['result'][0]['sex'] = ""
                    result['result'][1]['name'] = ""
                    result['result'][1]['sex'] = ""
                # img = vis_frame(orig_img, result, frame_count, debug, self.opt)
                # self.write_image(img, im_name, stream=stream if self.save_video else None)

                for i in range(len(result['result'])):
                    result['result'][i]['keypoints'] = result['result'][i]['keypoints'].tolist()
                    result['result'][i]['kp_score'] = result['result'][i]['kp_score'].tolist()
                    result['result'][i]['proposal_score'] = result['result'][i]['proposal_score'].tolist()

                result['debug'] = debug
                info.append(result)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, debug):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, debug))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


if __name__ == "__main__":
    sys.path.append("/home/dell/wz/AlphaPose/server")
    from server.req import return_name

    data = return_name()
    print(data)
    data_0 = data['0']
    print(data_0)
