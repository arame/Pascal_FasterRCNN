
###############################################   LAB 5   #########################################
# Code adapted from Lab 5 script written by Dr Alex Ter-Sarkisov@City, University of London, 2021 #
###################################################################################################
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
import matplotlib.pyplot as plt
from config import Hyper, Constants
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils import load_checkpoint, save_checkpoint

def train():
    # convert list to dict
    pascal_voc_classes = {}
    for id, name in enumerate(Hyper.pascal_categories):
        pascal_voc_classes[name] = id

    pascal_voc_classes_name = {v: k for k, v in pascal_voc_classes.items()} 
    num_classes = len(pascal_voc_classes)
    print(pascal_voc_classes, num_classes)
    # Modeling exercise: train fcn on Pascal VOC

    instance_data_args= {'classes':pascal_voc_classes, 'img_max_size':Hyper.img_max_size, 'dir':Constants.dir_images, 'problem':Hyper.pascal_problem, 'dir_label_bbox': Constants.dir_label_bbox}
    instance_data_point = dataset.PascalVOC2012Dataset(**instance_data_args)
    instance_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':True}
    instance_dataloader = data.DataLoader(instance_data_point, **instance_dataloader_args)

    fasterrcnn_args = {'box_score_thresh':Hyper.box_score_thresh, 'num_classes':91, 'min_size':512, 'max_size':800}
    # fasterrcnn_resnet50_fpn is pretrained on Coco's 91 classes
    # Pascal has 21 classes and this means pretrained has to be set to false
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,**fasterrcnn_args)
    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr': Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()), **fasterrcnn_optimizer_pars)
    #####################################################################
    if Constants.load_model:
        step = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer)

    fasterrcnn_model.train()  
    start_time = time.time()
    epoch = 0
    total_steps = 0
    total_loss = 0
    length_dataloader = len(instance_dataloader)
    for _ in range(Hyper.total_epochs):
        fasterrcnn_model.train()    # Set model to training mode
        epoch += 1
        epoch_loss = 0
        print(f"Starting epoch: {epoch}")
        step = 0
        for id, batch in enumerate(instance_dataloader):
            fasterrcnn_optimizer.zero_grad()
            _,X,y = batch
            total_steps += 1
            step += 1
            if step % 100 == 0:
                print(f"epoch: {epoch} step: {step} loss: {total_loss}")
            X,y['labels'],y['boxes'] = X.to(Constants.device), y['labels'].to(Constants.device), y['boxes'].to(Constants.device)
            # list of images
            images = [im for im in X]
            targets = []
            lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0)}
            # THIS IS IMPORTANT!!!!!
            # get rid of the first dimension (batch)
            # IF you have >1 images, make another loop
            # REPEAT: DO NOT USE BATCH DIMENSION 
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

                epoch_loss += total_loss.item()
                total_loss.backward()
                fasterrcnn_optimizer.step()
        epoch_loss = epoch_loss / length_dataloader
        print(f"Loss in epoch {epoch} = {epoch_loss}")
        # fasterrcnn_model.eval()     # Set model to validation mode
        if Constants.save_model:
            checkpoint = {
                "state_dict": fasterrcnn_model.state_dict(),
                "optimizer": fasterrcnn_optimizer.state_dict(),
                "epoch": epoch
            }
            save_checkpoint(checkpoint)


def check_if_target_bbox_degenerate(targets):

    if targets is None:
        return False

    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = None
        if len(boxes.shape) != 2:
            return True

        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = T.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb = boxes[bb_idx].tolist()
            print("All bounding boxes should have positive height and width.")
            print(f"Found invalid box {degen_bb} for target at index {target_idx}.")
            return True
    return False

 
if __name__ == "__main__":
    train()

