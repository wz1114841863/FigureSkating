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
from .vis_one_people import draw_one_people

import redis
sys.path.append("/home/dell/wz/AlphaPose/server")
from server.req import return_name
from server.skating2.redis_base.cache_redis import wirte_into_queue, get_from_queue, list_is_empty, redis_pool
redis_conn = redis.Redis(connection_pool= redis_pool)

ensumble = False

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
        self.input_3d = []

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

    def kal(self, coor, kalman):
        current_measurement = np.array([[np.float32(coor[0])], [np.float32(coor[1])]])
        kalman.correct(current_measurement)
        current_prediction = kalman.predict()
        return [current_prediction[0].item(), current_prediction[1].item()]

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
        frame_count = 0
        pose_count = 0
        info = []
        # ori_capture = cv2.VideoCapture(self.opt.video)
        # frame_height = int(ori_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(ori_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_fps = int(ori_capture.get(cv2.CAP_PROP_FPS))
        # frame_count_source = int(ori_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # ori_capture.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # kps_video_path = "/home/dell/wz/AlphaPose/examples/output/kps/kps_video.mp4"
        # kps_video_result = cv2.VideoWriter(kps_video_path, fourcc, frame_fps, (frame_width, frame_height), True)
        # keep looping infinitelyd
        while True:
            frame_count += 1
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, debug) = self.wait_and_get(self.result_queue)
            # 测试输出保存路径 
            img_name = str(frame_count) + ".jpg"
            img_path = "/home/dell/wz/AlphaPose/examples/output/test_output/" + img_name
        
            if orig_img is None:
                frame_count -= 1
                # if the thread indicator variable is set (img is None), stop the thread
                # if self.save_video:
                #     stream.release()
                # write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                # print("Results have been written to json.")
                
                #kps_video_result.release()
                print(len(final_result))
                print(len(self.input_3d))
                print(f"视频一共有{frame_count}帧")
                return

            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            # 未知bug，如果不保存再读取的话就会报错。
            # cv2.imwrite("./test.jpg", orig_img)
            # _img = cv2.imread("./test.jpg")

            if boxes is None or len(boxes) == 0:

                # if self.opt.save_img or self.save_video or self.opt.vis:
                #     # wz---begin
                #     from alphapose.utils.vis import vis_frame_empty
                #     img = vis_frame_empty(orig_img, frame_count)
                #     self.write_image(img, im_name, stream=stream if self.save_video else None)
                # bbox为空直接输出原始图片
                # cv2.imwrite(img_path, _img)
                # self.input_3d.append([{
                #             'bbox': [0, 0, 0, 0, 0],
                #             'keypoints': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                #                         [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                #                         [0,0,0]],  # [17, 2]
                #             'area': 0,
                #             'track_id': 0
                #         }])
                
                self.input_3d.append(np.zeros((17,2)))
                if ensumble == True:
                    wirte_into_queue(redis_conn, 'input_3d', np.zeros((17,2)).tolist())
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
                # import pdb; pdb.set_trace()
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox,
                                                                    hm_shape=hm_size,
                                                                    norm_type=norm_type)
                    # pose_coord: [17, 2], pose_score: [17, 1]
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))  
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))  
                preds_img = torch.cat(pose_coords)  # [1, 17, 2]
                preds_scores = torch.cat(pose_scores)  # [1, 17, 1]

                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)
                #import pdb;pdb.set_trace()
                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints': preds_img[k],  # [17, 2]
                            'kp_score': preds_scores[k],  # [17, 1]
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(
                                preds_scores[k]),
                            'idx': ids[k],
                            'box': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                        }
                    )
                    keypoints = []
                    for key in range(len(preds_img[k])):
                        keypoints.append([preds_img[k][key][0].item(), preds_img[k][key][1].item()
                                        #, preds_scores[k][key].item()
                                        ])
                    # self.input_3d.append([{
                    #         'bbox': [boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3], scores[k].item()],
                    #         'keypoints': keypoints,  # [17, 2]
                    #         'area': (boxes[k][2] - boxes[k][0])*(boxes[k][3] - boxes[k][1]),
                    #         'track_id': 0
                    #     }])
                    self.input_3d.append(preds_img[k].cpu().detach().numpy())
                    if ensumble == True:
                        wirte_into_queue(redis_conn, 'input_3d', preds_img[k].cpu().detach().numpy().tolist())
                result = {
                    'imgname': im_name,
                    'result': _result
                }
                # wz, draw img and save 
                result_img, angle = draw_one_people(orig_img.copy(), result)
                pose_count += 1
                if pose_count == 25:
                    if ensumble == True:
                        irte_into_queue(redis_conn, 'pose_queue', angle)
                    pose_count = 0
                result_img = cv2.resize(result_img,(640,480))
                ret, jpeg = cv2.imencode('.jpg', result_img)
                if ensumble == True:
                    wirte_into_queue(redis_conn, 'main_video_stream', jpeg.tobytes())
                #cv2.imwrite(img_path, result_img)
                # kps_video_result.write(result_img)
                # kps_video_result.write(orig_img.copy())
                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']
                final_result.append(result)
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

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, debug=None):
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
