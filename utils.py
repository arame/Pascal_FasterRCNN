import torch as T
import torchvision.transforms as transforms
from PIL import Image
from config import Constants
import os

def check_folder(folder):
    if os.path.isdir(folder) == False:
        os.mkdir(folder)


def checkfolders():
    # Make sure the output folders exist
    check_folder(Constants.backup_model_folder)
    

def save_checkpoint(checkpoint):
    check_folder(Constants.backup_model_folder)
    epoch = checkpoint["epoch"]
    print(f"=> Saving checkpoint for epoch {epoch}")
    T.save(checkpoint, Constants.backup_model_path)


def load_checkpoint(model, optimizer):
    checkpoint = T.load(Constants.backup_model_path)
    epoch = checkpoint["epoch"]
    print(f"=> Loading checkpoint from epoch {epoch}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return epoch