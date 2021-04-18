import torch as T
from config import OutputStore


def save_checkpoint(checkpoint):
    OutputStore.check_folder(OutputStore.backup_pascal_model_folder)
    epoch = checkpoint["epoch"]
    print(f"=> Saving checkpoint for epoch {epoch}")
    T.save(checkpoint, OutputStore.backup_model_path)


def load_checkpoint(model, optimizer):
    checkpoint = T.load(OutputStore.backup_model_path)
    epoch = checkpoint["epoch"]
    print(f"=> Loading checkpoint from epoch {epoch}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return epoch
