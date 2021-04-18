import torch as T
from config import OutputStore


def save_checkpoint(checkpoint):
    OutputStore.check_folder(OutputStore.backup_pascal_model_folder)
    epoch = checkpoint["epoch"]
    print(f"=> Saving checkpoint for epoch {epoch}")
    T.save(checkpoint, OutputStore.backup_model_path)


def load_checkpoint(model, optimizer, epoch):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    checkpoint = T.load(OutputStore.backup_model_path, checkpoint)
    epoch = checkpoint["epoch"]
    print(f"=> Loading checkpoint from epoch {epoch}")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model


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
