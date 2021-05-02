import torchvision
import torch
import numpy as np
import os,sys,re
from config import Constants, Hyper
from prediction_buffer import PredictionBuffer
from class_results_buffer import ClassResultsBuffer


# this method returns the size of the object inside the bounding box:
# input is a list in format xmin,ymin, xmax,ymax
def get_size_boxes(bboxes):
    _h, _w = bboxes[:,3] - bboxes[:,1], bboxes[:,2]-bboxes[:,0]
    sizes = _h *_w
    return sizes


# one gt for all predicted boxes
def box_iou(gt_box, pred_boxes, gt_box_size, pred_boxes_sizes):
    y1 = torch.max(gt_box[1], pred_boxes[:, 1])
    y2 = torch.min(gt_box[3], pred_boxes[:, 3])
    x1 = torch.max(gt_box[0], pred_boxes[:, 0])
    x2 = torch.min(gt_box[2], pred_boxes[:, 2])
    x_zeros = torch.zeros(x1.size()[0], device=Constants.device)
    y_zeros = torch.zeros(y1.size()[0], device=Constants.device)
    intersections = torch.max(x2 - x1, x_zeros) * torch.max(y2 - y1, y_zeros)
    union = gt_box_size+pred_boxes_sizes-intersections
    overlaps = intersections/union
    return overlaps


def compute_overlaps_boxes(bboxes1, bboxes2):
    s1 = get_size_boxes(bboxes1)
    s2 = get_size_boxes(bboxes2)
    iou_matrix = torch.zeros(bboxes1.size()[0], bboxes2.size()[0])
    for b in range(iou_matrix.size()[1]):
        gt_box = bboxes2[b]
        iou_matrix[:,b] = box_iou(gt_box, bboxes1, s2[b], s1)
    return iou_matrix


def compute_matches(buffer):
    # Sort predictions by score from high to low
    pred_boxes, pred_class_ids, pred_scores, gt_boxes, gt_class_ids = buffer.get_sorted_items()
    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_boxes(pred_boxes, gt_boxes)
    # separate predictions for each gt object (a total of gt_boxes splits
    split_overlaps = overlaps.t().split(1)
    # At the start all predictions are False Positives, all gts are False Negatives
    pred_match = torch.tensor([-1]).expand(pred_boxes.size()[0]).float()
    gt_match = torch.tensor([-1]).expand(gt_boxes.size()[0]).float()
    # Alex: loop through each column (gt object), get
    for _i, splits in enumerate(split_overlaps):
        # ground truth class
        gt_class = gt_class_ids[_i]
        if (splits > Hyper.box_score_thresh).any():
            # get best predictions, their indices in the IoU tensor and their classes
            global_best_preds_inds = torch.nonzero(splits[0] > Hyper.box_score_thresh).view(-1)
            pred_classes = pred_class_ids[global_best_preds_inds]
            best_preds = splits[0][splits[0] > Hyper.box_score_thresh]
            #  sort them locally-nothing else,
            local_best_preds_sorted = best_preds.argsort().flip(dims=(0,))
            # loop through each prediction's index, sorted in the descending order
            for p in local_best_preds_sorted:
                if pred_classes[p]==gt_class:
                    pred_match[global_best_preds_inds[p]] = _i
                    gt_match[_i] = global_best_preds_inds[p]
                    # important: if the prediction is True Positive, finish the loop
                    break

    return gt_match, pred_match, overlaps


# AP for a single IoU threshold and 1 image
def compute_ap(buffer):
    gt_match, pred_match, overlaps = compute_matches(buffer)
    # Compute precision and recall at each prediction box step
    precisions = (pred_match>-1).cumsum(dim=0).float().div(torch.arange(pred_match.numel()).float()+1)
    recalls = (pred_match>-1).cumsum(dim=0).float().div(gt_match.numel())
    # Pad with start and end values to simplify the math
    precisions = torch.cat([torch.tensor([0]).float(), precisions, torch.tensor([0]).float()])
    recalls = torch.cat([torch.tensor([0]).float(), recalls, torch.tensor([1]).float()])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    # Compute mean AP over recall range
    indices = torch.nonzero(recalls[:-1] !=recalls[1:]).squeeze_(1)+1
    MAP = torch.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return MAP, precisions, recalls, overlaps, gt_match, pred_match


def compute_class_ap(gt_match, pred_match):
    # Calculations at class level
    gt_match = torch.FloatTensor(gt_match)
    pred_match = torch.FloatTensor(pred_match)
    precisions = (pred_match>-1).cumsum(dim=0).float().div(torch.arange(pred_match.numel()).float()+1)
    recalls = (pred_match>-1).cumsum(dim=0).float().div(gt_match.numel())
    # Pad with start and end values to simplify the math
    precisions = torch.cat([torch.tensor([0]).float(), precisions, torch.tensor([0]).float()])
    recalls = torch.cat([torch.tensor([0]).float(), recalls, torch.tensor([1]).float()])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    # Compute mean AP over recall range
    indices = torch.nonzero(recalls[:-1] !=recalls[1:]).squeeze_(1)+1
    MAP = torch.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return MAP, precisions, recalls

