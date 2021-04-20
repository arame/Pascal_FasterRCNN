import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from config import Constants, Hyper


def get_model():
    fasterrcnn_args = {'box_score_thresh': Hyper.box_score_thresh, 'num_classes': 91, 'min_size': 512, 'max_size': 800}
    # fasterrcnn_resnet50_fpn is pretrained on Coco's 91 classes
    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **fasterrcnn_args)
    in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, Hyper.num_classes)
    fasterrcnn_model = fasterrcnn_model.to(Constants.device)
    fasterrcnn_optimizer_pars = {'lr': Hyper.learning_rate}
    fasterrcnn_optimizer = optim.Adam(list(fasterrcnn_model.parameters()), **fasterrcnn_optimizer_pars)
    return fasterrcnn_model, fasterrcnn_optimizer
