import numpy as np
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

class PredictionBuffer():
    def __init__(self, targets, predictions):
        p = predictions[0]
        t = targets[0]
        self.pred_boxes = p["boxes"]
        self.pred_class_ids = p["labels"]
        self.pred_scores = p["scores"]
        self.gt_boxes = t["boxes"]
        self.gt_class_ids = t["labels"]

    def get_items(self, i):
        return self.pred_boxes[i], self.pred_class_ids[i], self.pred_scores[i], self.gt_boxes, self.gt_class_ids
        # pred_boxes, pred_class_ids, pred_scores, gt_boxes, gt_class_ids
