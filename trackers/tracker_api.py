# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
#
# -----------------------------------------------------

"""API of tracker"""
import os
import sys
import cv2 as cv

sys.path.insert(0, os.path.dirname(__file__))
from collections import deque
import torch.nn as nn

from utils.utils import *
from utils.log import logger
from utils.kalman_filter import KalmanFilter
from tracking.matching import *
from tracking.basetrack import BaseTrack, TrackState
from ReidModels.osnet_ain import osnet_ain_x1_0
from ReidModels.resnet_fc import resnet50_fc512

sys.path.append("..")
from server.req import send_image


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, pose, crop_box, file_name, ps, sc, ss, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat, 0)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.99
        self.pose = pose
        self.detscore = ps
        self.crop_box = crop_box
        self.file_name = file_name
        self.pose_coord = sc
        self.pose_score = ss
        self.feat_count = 0

    def update_features(self, feat, feat_choose):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        elif feat_choose == 1 or feat_choose == 0:
            alpha = 0.99
            self.feat_count += 1
            # if self.feat_count > 37:
            #     alpha = 1
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if self.next_id() == 1:
            self.track_id = 1
        else:
            self.track_id = 2
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, feat, new_id=False):

        self._tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(self._tlwh))

        self.update_features(new_track.curr_feat, feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.pose = new_track.pose
        self.detscore = new_track.detscore
        self.crop_box = new_track.crop_box
        self.file_name = new_track.file_name
        self.pose_coord = new_track.pose_coord
        self.pose_score = new_track.pose_score

    def update(self, new_track, frame_id, feat, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.pose = new_track.pose
        self.detscore = new_track.detscore
        self.crop_box = new_track.crop_box
        self.file_name = new_track.file_name
        self._tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(self._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.pose_coord = new_track.pose_coord
        self.pose_score = new_track.pose_score
        if update_feature:
            self.update_features(new_track.curr_feat, feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Tracker(object):
    def __init__(self, opt, args):
        self.opt = opt
        self.num_joints = 17
        self.frame_rate = opt.frame_rate
        # m = ResModel(n_ID=opt.nid)
        if self.opt.arch == "res50-fc512":
            m = resnet50_fc512(num_classes=1, pretrained=False)
        elif self.opt.arch == "osnet_ain":
            m = osnet_ain_x1_0(num_classes=1, pretrained=False)

        self.model = nn.DataParallel(m, device_ids=args.gpus).to(args.device).eval()

        load_pretrained_weights(self.model, self.opt.loadmodel)
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(self.frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

        self.choose = 0  # 0 继续选择 1 停止选择 可用status替代
        self.feat = -1  # -1一个box，更第一次  0 2个box iou小 更新 1 2个box iou小不更新

        # 0.未有二人  1.一个box且相交  2.一个box且不相交  3.有两个box  4. 全lost
        self.status = 0
        self.debug = [0, 0, 0, 0, 0]
        self.img2 = 0
        self.reidimg = args.reidimg
        # 0 没遇到onetwo 1 
        self.onetwo = 0

    def update(self, img0, inps=None, bboxs=None, pose=None, cropped_boxes=None, file_name='', pscores=None,
               pose_coords=None, pose_scores=None,
               _debug=False):
        # bboxs:[x1,y1.x2,y2]
        self.frame_id += 1
        # if self.frame_id == 300:
        #     if self.frame_id == 631:
        #         pass

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        ''' Step 1: Network forward, get human identity embedding'''
        assert len(inps) == len(bboxs), 'Unmatched Length Between Inps and Bboxs'
        assert len(inps) == len(pose), 'Unmatched Length Between Inps and Heatmaps'
        with torch.no_grad():
            feats = self.model(inps).cpu().numpy()
        bboxs = np.asarray(bboxs)
        if len(bboxs) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:]), 0.9, f, p, c, file_name, ps, sc, ss, 30) for
                          (tlbrs, f, p, c, ps, sc, ss) in
                          zip(bboxs, feats, pose, cropped_boxes, pscores, pose_coords, pose_scores)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        detections = pre_pose(detections, strack_pool)
        # detections = pre_pose(detections, tracked_stracks)

        # status
        if len(strack_pool) == 2:
            if len(tracked_stracks) == 1:
                if boxiou(strack_pool[0].tlwh, strack_pool[1].tlwh) > 0:
                    self.status = 1
                else:
                    self.status = 2
            elif len(tracked_stracks) == 2:
                self.status = 3
            else:
                self.status = 4
        else:
            self.status = 0
        self.debug[3] = self.status

        # 1-2 embedding
        # if not (self.status == 1 and len(detections) == 2):
        #     self.onetwo = 0
        # else:
        #     self.onetwo = 1

        # if self.onetwo == 1:
        #     if boxiou(detections[0]._tlwh, detections[1]._tlwh):

        if not (self.status == 1 and len(detections) == 2):
            pose_thres = 30
        else:
            pose_thres = 5
        # pose_thres = 15
        # if False:

        # pose
        dists_pose = pose_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists_pose, thresh=pose_thres)
        self.debug[0] += len(matches)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # if len(tracked_stracks) == 1 and len(self.lost_stracks) == 2 \
                #     and len(detections) == 2:
                #     # pass#特征 795.jpg
                #     dists_emb = embedding_distance(strack_pool, detections)
                #     match, un_track, un_detection = linear_assignment(dists_emb, thresh=0.8)
                #     if len(match) == 2:
                #         break
                track.update(det, self.frame_id, self.feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, self.feat, new_id=False)
                refind_stracks.append(track)

        # embedding
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track]
        strack_pool = r_tracked_stracks
        dists_emb = embedding_distance(strack_pool, detections)
        # self.debug[4] = dists_emb
        # dists_emb = fuse_motion(self.kalman_filter, dists_emb, strack_pool, detections)
        # matches, u_track, u_detection = linear_assignment(dists_emb, thresh=0.7)
        matches, u_track, u_detection = linear_assignment(dists_emb, thresh=0.8)
        self.debug[1] += len(matches)

        for itracked, idet in matches:
            # track = strack_pool[itracked]
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, self.feat, new_id=False)
                refind_stracks.append(track)

        # # pose+emb
        # dists_emb = embedding_distance(strack_pool, detections)
        # if len(self.lost_stracks) != 2:
        #     dists_pose = pose_distance(strack_pool, detections)
        #     dists_emb = dists_pose/1000 + dists_emb
        # matches, u_track, u_detection = linear_assignment(dists_emb, thresh=100)
        # self.debug[4] = dists_emb
        # self.debug[0] += len(matches)

        # for itracked, idet in matches:
        #     track = strack_pool[itracked]
        #     det = detections[idet]
        #     if track.state == TrackState.Tracked:
        #         # if len(tracked_stracks) == 1 and len(self.lost_stracks) == 2 \
        #         #     and len(detections) == 2:
        #         #     # pass#特征 795.jpg
        #         #     dists_emb = embedding_distance(strack_pool, detections)
        #         #     match, un_track, un_detection = linear_assignment(dists_emb, thresh=0.8)
        #         #     if len(match) == 2:
        #         #         break
        #         track.update(det, self.frame_id, self.feat)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, self.feat, new_id=False)
        #         refind_stracks.append(track)

        # Step 3: Second association, with IOU
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_iou = iou_distance(r_tracked_stracks, detections)
        # matches, u_track, u_detection = linear_assignment(dists_iou, thresh=0.5)
        matches, u_track, u_detection = linear_assignment(dists_iou, thresh=0.5)
        self.debug[2] += len(matches)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.feat)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, self.feat, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, self.feat)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # tmp_active_track = joint_stracks(self.tracked_stracks, activated_starcks)
        # tmp_active_track = joint_stracks(tmp_active_track, refind_stracks)

        # tmp_lost_track = sub_stracks(self.lost_stracks, tmp_active_track)
        # tmp_lost_track.extend(lost_stracks)

        # tmp_active_track, tmp_lost_track = remove_duplicate_stracks(tmp_active_track, tmp_lost_track)
        # num = len(joint_stracks(tmp_active_track, tmp_lost_track))

        if self.choose != 1:
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

        """ Step 5: Update state"""
        # for track in self.lost_stracks:
        #     if self.frame_id - track.end_frame > self.max_time_lost:
        #         track.mark_removed()
        #         removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        # self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        if len([track for track in self.tracked_stracks if track.is_activated]) == 2:
            self.choose = 1
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks]

        # feat
        num_track = len([track for track in self.tracked_stracks if track.is_activated])
        if num_track == 2:
            if boxiou(self.tracked_stracks[0].tlwh, self.tracked_stracks[1].tlwh) < 0.2:
                self.feat = 0
                if self.img2 == 0 and boxiou(self.tracked_stracks[0].tlwh, self.tracked_stracks[1].tlwh) == 0:
                    self.img2 = 1
                    img_add = []
                    print(self.frame_id)
                    if not os.path.exists(self.reidimg):
                        os.makedirs(self.reidimg)
                    for t in self.tracked_stracks:
                        tlwh = [int(x) for x in t._tlwh]
                        tlbr = [tlwh[1], tlwh[0], tlwh[1] + tlwh[3], tlwh[0] + tlwh[2]]
                        # print(t._tlwh, t.crop_box, [x.shape for x in inps])
                        crop_img = img0[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3], [2, 1, 0]]
                        cv.imwrite(self.reidimg + str(t.track_id) + '.jpg', crop_img)
                        img_add.append('/home/dell/first/AlphaPose/' + self.reidimg + str(t.track_id) + '.jpg')
                    # cv.imwrite('/home/dell/first/AlphaPose/examples/demo/output/id/0.jpg', img0[:,:,[2,1,0]])
                    # if len(img_add) == 2:
                    #     # pass
                    #     send_image(img_add[0], img_add[1])
                    #     # print(f"这里应该发送图片。")
            else:
                self.feat = 1
        else:
            self.feat = -1

        self.debug[4] = self.feat

        if _debug:
            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks


def pose_sim(pose1, pose2, thres=8):
    score = 0
    for key in range(17):
        if (pose1[key][0] - pose2[key][0]) ** 2 + (pose1[key][1] - pose2[key][1]) ** 2 < thres ** 2:
            score += 1
    return score


def pose_score1(track1, track2):
    score = 0
    thres = 0.6
    pose1 = track1.pose_coord
    pose2 = track2.pose_coord
    score1 = track1.pose_score
    score2 = track2.pose_score
    idx = []
    for key in range(17):
        if score1[key] > thres and score2[key] > thres:
            idx.append(key)
    if len(idx) < 5:
        return 10000
    for key in idx:
        score += ((pose1[key][0] - pose2[key][0]) ** 2 + (pose1[key][1] - pose2[key][1]) ** 2) ** 0.5
    return score / len(idx)


def pose_distance(tracks, detections):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    for t in range(len(tracks)):
        for d in range(len(detections)):
            # cost_matrix[t, d] = 17 - pose_sim(tracks[t].pose_coord, detections[d].pose_coord, thres=40)
            cost_matrix[t, d] = pose_score1(tracks[t], detections[d])
    return cost_matrix


def dedup_pose(tlist):
    if len(tlist) < 2:
        return tlist
    remove = []
    len_t = len(tlist)
    for i1 in range(len_t):
        for i2 in range(len_t):
            if i2 > i1:
                if pose_sim(tlist[i1].pose_coord, tlist[i2].pose_coord) >= 3:
                    if tlist[i1].pose_score.sum() > tlist[i2].pose_score.sum():
                        remove.append(tlist[i2])
                    else:
                        remove.append(tlist[i1])
    remove = list(set(remove))
    for re in remove:
        tlist.remove(re)
    return tlist


def low_del(tlist):
    remove = []
    for i in range(len(tlist)):
        # if tlist[i].detscore < 0.4 and tlist[i].pose_score.sum() < 6:
        if tlist[i].pose_score.sum() < 10:
            if tlist[i].detscore < 0.7:
                remove.append(tlist[i])
    for re in remove:
        tlist.remove(re)
    return tlist


def pick2(tlist):
    if len(tlist) < 2:
        return tlist
    remove = []
    tlist.sort(key=lambda x: x._tlwh[2] * x._tlwh[3])
    area = [x._tlwh[2] * x._tlwh[3] for x in tlist]
    # for i in range(len(area)):
    #     if area[i] < area[-1] * 0.4:
    #         remove.append(tlist[i])
    # for re in remove:
    #     tlist.remove(re)
    if len(tlist) > 1:
        tlist = tlist[-2:]
    return tlist


def de_jump(tlist, track):
    if len(tlist) == 0:
        return []
    remove = []
    if len(track) == 2:
        center0 = track[0].mean
        center1 = track[1].mean
        center = [center0, center1]
        for t in tlist:
            re_num = 0
            for c in center:
                if ((t._tlwh[0] + t._tlwh[2] / 2 - c[0]) ** 2 +
                    (t._tlwh[1] + t._tlwh[3] / 2 - c[1]) ** 2) ** 0.5 < 200:
                    re_num = 1
            if re_num == 0:
                remove.append(t)
    if len(track) == 1:
        center0 = track[0].mean
        center = [center0, center0]
        for t in tlist:
            re_num = 0
            for c in center:
                if ((t._tlwh[0] + t._tlwh[2] / 2 - c[0]) ** 2 +
                    (t._tlwh[1] + t._tlwh[3] / 2 - c[1]) ** 2) ** 0.5 < 200:
                    re_num = 1
            if re_num == 0:
                remove.append(t)
    for re in remove:
        tlist.remove(re)
    return tlist


def pre_pose(tlist, track):
    tlist = low_del(tlist)
    tlist = dedup_pose(tlist)
    tlist = de_jump(tlist, track)
    tlist = pick2(tlist)
    return tlist


# 合并两个关节点列表
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


# box: [x, y, w, h]
def point_in_box(point, box):
    if (box[0] < point[0] < box[0] + box[2]) and (box[1] < point[1] < box[0] + box[3]):
        return True
    else:
        return False


def boxiou(box1, box2):
    x_list = [box1[0], box1[0] + box1[2], box2[0], box2[0] + box2[2]]
    y_list = [box1[1], box1[1] + box1[3], box2[1], box2[1] + box2[3]]
    x_list.sort()
    y_list.sort()

    # 求包裹面积
    wrap_area = (x_list[3] - x_list[0]) * (y_list[3] - y_list[0])
    cross_point = [(x_list[2] + x_list[1]) / 2, (y_list[2] + y_list[1]) / 2]
    # 判断中心点是否都在两个box内
    if point_in_box(cross_point, box1) and point_in_box(cross_point, box2):
        cross_area = (x_list[2] - x_list[1]) * (y_list[2] - y_list[1])
        score = cross_area / wrap_area
    else:
        score = 0
    return score


# 删除列表a中与列表b重复的关节点
def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


#
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
