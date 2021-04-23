import numpy as np
import os, sys, re
import time
import torch
import cv2
from pascal_data import PascalVOC2012Dataset
from config import Hyper, Constants, OutputStore
from utils import load_checkpoint, check_if_target_bbox_degenerate
from metrics import compute_ap
from prediction_buffer import PredictionBuffer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test_for_class(fasterrcnn_model):
    OutputStore.check_folder(OutputStore.dir_output_test_images)
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print("-" * 100)
    print(f"{start_time} Starting testing the model")
    test_dataloader = PascalVOC2012Dataset.get_data_loader(Constants.dir_test_images, "test")
    fasterrcnn_model.eval()  # Set to eval mode for validation
    step = 0
    tot_MAP = 0
    for id, batch in enumerate(test_dataloader):
        _, X, img, img_file, y = batch
        step += 1
        if step % 100 == 0:
            curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
            print(f"-- {curr_time} step: {step}")
        X, y['labels'], y['boxes'] = X.to(Constants.device), y['labels'].to(Constants.device), y['boxes'].to(
            Constants.device)
        img_file = img_file[0]
        img = np.squeeze(img)
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
        buffer = PredictionBuffer(targets, predictions)
        MAP, precisions, recalls, overlaps = compute_ap(buffer)
        tot_MAP += MAP

    ave_MAP = tot_MAP / step
    print(f"Average MAP = {ave_MAP}")


if __name__ == "__main__":
    epoch = Hyper.total_epochs
    model, _ = load_checkpoint(epoch)
    test_for_class(model)
