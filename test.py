import numpy as np
import os, sys, re
import time
import torch
from pascal_data import PascalVOC2012Dataset
from config import Hyper, Constants
from utils import load_checkpoint, save_checkpoint, check_if_target_bbox_degenerate
from metrics import compute_ap
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test(fasterrcnn_model):
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print("-" * 100)
    print(f"{start_time} Starting testing the model")
    instance_dataloader = PascalVOC2012Dataset.get_data_loader(Constants.dir_test_images)
    fasterrcnn_model.eval()  # Set to eval mode for validation
    step = 0
    tot_MAP = 0
    tot_overlaps = 0
    tot_overlaps_cnt = 0
    for id, batch in enumerate(instance_dataloader):
        _, X, y = batch
        step += 1
        if step % 100 == 0:
            curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
            print(f"-- {curr_time} step: {step}")
        X, y['labels'], y['boxes'] = X.to(Constants.device), y['labels'].to(Constants.device), y['boxes'].to(
            Constants.device)
        # list of images
        images = [im for im in X]
        targets = []
        lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0)}
        targets.append(lab)
        is_bb_degenerate = check_if_target_bbox_degenerate(targets)
        if is_bb_degenerate:
            continue  # Ignore images with degenerate bounding boxes
        # avoid empty objects
        if len(targets) == 0:
            continue

        # Get the predictions from the trained model
        predictions = fasterrcnn_model(images, targets)
        # predictions = predictions.to(Constants.model)
        # now compare the predictions with the ground truth values in the targets

        MAP, precisions, recalls, overlaps = compute_ap(predictions, targets)
        # print(f"map: {MAP}, precisions: {precisions}, recalls: {recalls}, overlaps: {overlaps}")
        tot_MAP += MAP
        #tot_overlaps += torch.sum(overlaps)
        #tot_overlaps_cnt += len(overlaps[0])

    ave_MAP = tot_MAP / step
    print(f"Average MAP = {ave_MAP}")


if __name__ == "__main__":
    epoch = Hyper.total_epochs
    model = load_checkpoint(epoch)
    test(model)
