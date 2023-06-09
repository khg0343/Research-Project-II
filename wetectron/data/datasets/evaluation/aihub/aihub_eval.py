from __future__ import division

import os
from collections import defaultdict
import numpy as np
from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import boxlist_iou


def do_aihub_evaluation(dataset, predictions, output_folder, logger):
    pred_boxlists = []
    gt_boxlists = []
    for index, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(index)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(index)
        gt_boxlists.append(gt_boxlist)
        
    result = eval_detection_aihub(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_detection_aihub(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_aihub_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_aihub_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_aihub_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_occluded = gt_boxlist.get_field("occluded").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_occluded_l = gt_occluded[gt_mask_l]

            n_pos[l] += np.logical_not(gt_occluded_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_occluded_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_aihub_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
