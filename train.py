
###############################################   LAB 5   #########################################
# Code adapted from Lab 5 script written by Dr Alex Ter-Sarkisov@City, University of London, 2021 #
###################################################################################################
import numpy as np
import os,sys,re
import time
import torch
from model import get_model
from pascal_data import PascalVOC2012Dataset
from metrics import compute_ap
from config import Hyper, Constants
from utils import load_checkpoint, save_checkpoint, check_if_target_bbox_degenerate
from results import save_loss_per_epoch_chart, save_ave_MAP_per_epoch_chart, save_ave_overlaps_per_epoch_chart
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(epoch = 0):
    # convert list to dict
    pascal_voc_classes = {}
    for id, name in enumerate(Hyper.pascal_categories):
        pascal_voc_classes[name] = id

    pascal_voc_classes_name = {v: k for k, v in pascal_voc_classes.items()} 
    print(pascal_voc_classes, Hyper.num_classes)
    # Modeling exercise: train fcn on Pascal VOC
    train_dataloader = PascalVOC2012Dataset.get_data_loader(Constants.dir_images)
    val_dataloader = PascalVOC2012Dataset.get_data_loader(Constants.dir_test_images)

    fasterrcnn_model, fasterrcnn_optimizer = get_model()

    print(fasterrcnn_model)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    #####################################################################
    if Constants.load_model:
        fasterrcnn_model = load_checkpoint(fasterrcnn_model, fasterrcnn_optimizer, epoch)

    start_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"{start_time} Starting epoch: {epoch}")
    total_steps = 0
    total_loss = 0
    length_dataloader = len(train_dataloader)
    loss_per_epoch = []
    ave_MAP_per_epoch = []
    ave_overlaps_per_epoch = []
    for _ in range(Hyper.total_epochs):
        fasterrcnn_model.train()
        epoch += 1
        epoch_loss = 0
        print(f"Starting epoch: {epoch}")
        step = 0
        for id, batch in enumerate(train_dataloader):
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

        fasterrcnn_model.eval()  # Set to eval mode for validation
        step = 0
<<<<<<< HEAD
        tot_MAP = 0
        ave_MAP = 0
        tot_overlaps = 0
        tot_overlaps_cnt = 0
        for id, batch in enumerate(val_dataloader):
            _, X, y = batch
            step += 1
                curr_time = time.strftime('%Y/%m/%d %H:%M:%S')
=======
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

            # now compare the predictions with the ground truth values in the targets
            MAP, precisions, recalls, overlaps = compute_ap(predictions, targets)
            # print(f"map: {mAP}, precisions: {precisions}, recalls: {recalls}, overlaps: {overlaps}")
            tot_MAP += MAP
            tot_overlaps += sum(overlaps)
            tot_overlaps_cnt += len(overlaps)

        ave_mAP = tot_mAP / step
        ave_mAP_per_epoch.append(ave_mAP)
    save_loss_per_epoch_chart(loss_per_epoch)
    save_ave_MAP_per_epoch_chart(ave_MAP_per_epoch)
    save_ave_overlaps_per_epoch_chart(ave_overlaps_per_epoch)
    end_time = time.strftime('%Y/%m/%d %H:%M:%S')
    print(f"Training end time: {end_time}")
    return fasterrcnn_model

 
if __name__ == "__main__":
    train()

