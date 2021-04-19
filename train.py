
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
from torch.utils import data
from config import Hyper, Constants
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils import load_checkpoint, save_checkpoint, check_if_target_bbox_degenerate
from results import save_loss_per_epoch_chart

def train(epoch = 0):
    # convert list to dict
    pascal_voc_classes = {}
    for id, name in enumerate(Hyper.pascal_categories):
        pascal_voc_classes[name] = id

    pascal_voc_classes_name = {v: k for k, v in pascal_voc_classes.items()} 
    num_classes = len(pascal_voc_classes)
    print(pascal_voc_classes, num_classes)
    # Modeling exercise: train fcn on Pascal VOC

    instance_data_args= {'classes':pascal_voc_classes, 
                        'img_max_size':Hyper.img_max_size, 
                        'dir':Constants.dir_images, 
                        'dir_label_bbox': Constants.dir_label_bbox}
    instance_data_point = dataset.PascalVOC2012Dataset(**instance_data_args)
    instance_dataloader_args = {'batch_size':Hyper.batch_size, 'shuffle':True}
    instance_dataloader = data.DataLoader(instance_data_point, **instance_dataloader_args)

    fasterrcnn_args = {'box_score_thresh':Hyper.box_score_thresh, 'num_classes':91, 'min_size':512, 'max_size':800}
    # fasterrcnn_resnet50_fpn is pretrained on Coco's 91 classes
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,**fasterrcnn_args)
    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr': Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()), **fasterrcnn_optimizer_pars)
    #####################################################################
    if Constants.load_model:
        fasterrcnn_model = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer, epoch)

    fasterrcnn_model.train()  
    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"{start_time} Starting epoch: {epoch}")
    total_steps = 0
    total_loss = 0
    length_dataloader = len(instance_dataloader)
    loss_per_epoch = []
    for _ in range(Hyper.total_epochs):
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
                curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
                print(f"-- {curr_time} epoch: {epoch} step: {step} loss: {total_loss}")

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
            if len(targets) == 0:
                continue    # Ignore if no targets

            loss = fasterrcnn_model(images, targets)
            total_loss = 0
            for k in loss.keys():
                total_loss += loss[k]

            epoch_loss += total_loss.item()
            total_loss.backward()
            fasterrcnn_optimizer.step()
        epoch_loss = epoch_loss / length_dataloader
        loss_per_epoch.append(epoch_loss)
        print(f"Loss in epoch {epoch} = {epoch_loss}")
        if Constants.save_model:
            checkpoint = {
                "state_dict": fasterrcnn_model.state_dict(),
                "optimizer": fasterrcnn_optimizer.state_dict(),
                "epoch": epoch
            }
            save_checkpoint(checkpoint)
    save_loss_per_epoch_chart(loss_per_epoch)
    end_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"Training end time: {end_time}")
    return fasterrcnn_model


 
if __name__ == "__main__":
    train()

