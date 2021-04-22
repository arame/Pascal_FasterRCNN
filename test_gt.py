import numpy as np
import os, sys, re
import time
import torch
import cv2
from pascal_data import PascalVOC2012Dataset
from config import Hyper, Constants, OutputStore
from utils import load_checkpoint, check_if_target_bbox_degenerate
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
''' 
    This code prints off the test images with the ground truth bounding boxes. 
    Only need to run this once as the model is not being tested here.
    Then the images can be compared with the predicted bounding boxes from the model for the same images. 
'''


def test_gt():
    OutputStore.check_folder(OutputStore.dir_output_test_images + "_gt")
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print("-" * 100)
    print(f"{start_time} Starting output of the test images ground truth")
    test_dataloader = PascalVOC2012Dataset.get_data_loader(Constants.dir_test_images, "test")
    step = 0
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

        output_annotated_images(targets, img, img_file)


def output_annotated_images(targets, img, img_file):
    boxes = targets[0]["boxes"]
    labels = targets[0]["labels"]

    # this will help us create a different colour for each class
    COLOURS = np.random.uniform(0, 255, size=(Hyper.num_classes, 3))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        label_index = labels[i]
        text = f"{Hyper.pascal_categories[label_index]}"
        color = COLOURS[labels[i]]
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(img, text, (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    lineType=cv2.LINE_AA)

    file_bb = img_file.replace(".jpg", f"_out_gt.jpg")
    path = os.path.join(OutputStore.dir_output_test_images + "_gt", file_bb)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    test_gt()
