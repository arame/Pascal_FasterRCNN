import numpy as np
import os, sys, re
import time
import torch
import cv2
from pascal_data import PascalVOC2012Dataset
from config import Hyper, Constants, OutputStore
from utils import load_checkpoint, check_if_target_bbox_degenerate
from metrics import compute_ap, compute_class_ap
from prediction_buffer import PredictionBuffer
from class_data import ClassData
from results import save_class_metrics
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test(fasterrcnn_model):
    OutputStore.check_folder(OutputStore.dir_output_test_images)
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print("-" * 100)
    print(f"{start_time} Starting testing the model")

    class_data = ClassData()
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
        
        MAP, precisions, recalls, overlaps, gt_match, pred_match = compute_ap(buffer)
        class_data.add_matches(buffer, gt_match, pred_match)

        # MAP, precisions, recalls, overlaps = compute_ap(predictions, targets)
        tot_MAP += MAP
        output_annotated_images(predictions, img, img_file)
        output_stats_for_images(MAP, precisions, recalls, overlaps, img_file)

    #output_stats_for_class(MAP, precisions, recalls, overlaps)
    print("Class level precision and recalls")
    print("---------------------------------")
    for i in range(Hyper.num_classes):
        class_name = Hyper.pascal_categories[i]
        class_gt_match = class_data.gt_match_dict[i]
        class_pred_match = class_data.pred_match_dict[i]
        if len(class_gt_match) > 0:
            MAP, precisions, recalls = compute_class_ap(class_gt_match, class_pred_match)
            if MAP == 0:
                continue

            save_class_metrics(class_name, precisions, recalls)
        print(f"MAP for {class_name}: {MAP}")
    print("---------------------------------\n\n")
    ave_MAP = tot_MAP / step
    print(f"Average MAP = {ave_MAP}")
    print("*** Test run completed ***")

def output_annotated_images(prediction, img, img_file):
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = torch.round(prediction[0]["scores"] * 10) / 10

    # this will help us create a different colour for each class
    COLOURS = np.random.uniform(0, 255, size=(Hyper.num_classes, 3))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    for j in range(2):
        for i, box in enumerate(boxes):
            label_index = labels[i]
            score = str(scores[i].item())[0:3]
            if j == 0:
                text = f"{Hyper.pascal_categories[label_index]}"
            else:
                text = f"{Hyper.pascal_categories[label_index]} {score}"
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

        file_bb = img_file.replace(".jpg", f"_out{j}.jpg")
        path = os.path.join(OutputStore.dir_output_test_images, file_bb)
        cv2.imwrite(path, img)


def output_stats_for_images(MAP, precisions, recalls, overlaps, img_file):

    txt_file = img_file.replace("jpg", "txt")
    path = os.path.join(OutputStore.dir_output_test_images, txt_file)
    lines = [f"{img_file} metrics",
             "-"*20, f"MAP: {MAP}",
             f"precisions: {precisions}",
             f"recalls: {recalls}",
             f"overlaps: {overlaps}"]
    with open(path, "w") as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    #epoch = Hyper.total_epochs
    epoch = 13
    model, _ = load_checkpoint(epoch)
    test(model)
