import numpy as np
import os, sys, re
import time
from model import get_model
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
    tot_mAP = 0
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
        # TODO IoU calculations and accuracy calculations

        mAP, precisions, recalls, overlaps = compute_ap(predictions, targets)
        # print(f"map: {mAP}, precisions: {precisions}, recalls: {recalls}, overlaps: {overlaps}")
        tot_mAP += mAP

    ave_mAP = tot_mAP / step
    print(f"Average mAP = {ave_mAP}")


if __name__ == "__main__":
    fasterrcnn_model, fasterrcnn_optimizer = get_model()
    epoch = Hyper.total_epochs
    model = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer, epoch)
    test(model)
