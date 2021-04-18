import numpy as np
import os,sys,re
import time
from PIL import Image as PILImage
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms as transforms
import pascal_data as dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils import data
from config import Hyper, Constants
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils import load_checkpoint, save_checkpoint, check_if_target_bbox_degenerate
from results import save_loss_per_epoch_chart


def test(fasterrcnn_model):
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print("-"*100)
    print(f"{start_time} Starting testing the model")
    pascal_voc_classes = {}
    for id, name in enumerate(Hyper.pascal_categories):
        pascal_voc_classes[name] = id

    instance_data_args= {'classes':pascal_voc_classes, 
                        'img_max_size':Hyper.img_max_size, 
                        'dir':Constants.dir_test_images, 
                        'dir_label_bbox': Constants.dir_label_bbox}
    instance_data_point = dataset.PascalVOC2012Dataset(**instance_data_args)
    instance_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':False}
    instance_dataloader = data.DataLoader(instance_data_point, **instance_dataloader_args)
    length_dataloader = len(instance_dataloader)

    fasterrcnn_model.eval()     # Set to eval mode for validation
    total_loss = 0
    test_loss = 0
    step = 0
    for id, batch in enumerate(instance_dataloader):
        _,X,y = batch
        step += 1
        if step % 100 == 0:
            curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
            print(f"-- {curr_time} step: {step} loss: {total_loss}")
        X,y['labels'],y['boxes'] = X.to(Constants.device), y['labels'].to(Constants.device), y['boxes'].to(Constants.device)
        # list of images
        images = [im for im in X]
        targets = []
        lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0)}
        targets.append(lab)
        is_bb_degenerate = check_if_target_bbox_degenerate(targets)
        if is_bb_degenerate:
            continue  # Ignore images with degenerate bounding boxes
        # avoid empty objects
        if len(targets) > 0:
            loss = fasterrcnn_model(images, targets)
            total_loss = 0
            for k in loss.keys():
                total_loss += loss[k]
            
            test_loss += total_loss

        test_loss = test_loss / length_dataloader
        print(f"Loss from testing = {test_loss}")


if __name__ == "__main__":
    fasterrcnn_args = {'box_score_thresh':Hyper.box_score_thresh, 'num_classes':91, 'min_size':512, 'max_size':800}
    # fasterrcnn_resnet50_fpn is pretrained on Coco's 91 classes
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,**fasterrcnn_args)
    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr': Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()), **fasterrcnn_optimizer_pars)
    model = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer)
    test(model)