import numpy as np
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

class PredictionBuffer():
    def __init__(self, targets=None, predictions=None):
        if targets == None:
            self.gt_boxes = []
            self.gt_class_ids = []
        else:
            t = targets[0]
            self.gt_boxes = t["boxes"]
            self.gt_class_ids = t["labels"]

        if predictions == None:
            self.pred_boxes = []
            self.pred_class_ids = []
            self.pred_scores = []
        else:
            p = predictions[0]
            self.pred_boxes = p["boxes"]
            self.pred_class_ids = p["labels"]
            self.pred_scores = p["scores"]


    def get_sorted_items(self, i):
        i = self.pred_scores.argsort().flip(dims=(0,))
        return self.pred_boxes[i], self.pred_class_ids[i], self.pred_scores[i], self.gt_boxes, self.gt_class_ids

