import torch
from alphapose.utils.transforms import get_func_heatmap_to_coord


def track(cfg, tracker, args, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores):
    # 关节点重复计算，后续可以优化
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    pose_coords = []
    pose_scores = []
    for i in range(hm.shape[0]):
        bbox = cropped_boxes[i].tolist()
        pose_coord, pose_score = heatmap_to_coord(hm[i][EVAL_JOINTS], bbox, hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
                                                        norm_type=None)
        pose_coords.append(pose_coord)
        pose_scores.append((pose_score))

    hm = hm.cpu().data.numpy()
    online_targets = tracker.update(orig_img, inps, boxes, hm, cropped_boxes, im_name, scores, pose_coords, pose_scores, _debug=False)
    debug = tracker.debug
    new_boxes, new_scores, new_ids, new_hm, new_crop = [], [], [], [], []
    for t in online_targets:
        tlbr = t.tlbr
        tid = t.track_id
        thm = t.pose
        tcrop = t.crop_box
        tscore = t.detscore
        new_boxes.append(tlbr)
        new_crop.append(tcrop)
        new_hm.append(thm)
        new_ids.append(tid)
        new_scores.append(tscore)

    new_hm = torch.Tensor(new_hm).to(args.device)
    return new_boxes, new_scores, new_ids, new_hm, new_crop, debug
