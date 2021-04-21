from config import Hyper, Constants
from utils import load_checkpoint
import os
from PIL import Image as PILImage
import numpy as np
from torchvision import transforms as transforms


def individual_image(fasterrcnn_model, image):
    fasterrcnn_model.eval()
    prediction = fasterrcnn_model(image)
    print(prediction)


def transform_img(img):
    h, w, c = img.shape
    h_, w_ = Hyper.img_max_size[0], Hyper.img_max_size[1]
    img_size = tuple((h_, w_))
    # these mean values are for BGR!!

    t_ = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.407, 0.457, 0.485],
        #                     std=[1,1,1])])
    ])
    img = t_(img)
    img = img.unsqueeze_(0)
    img = img.to(Constants.device)
    # need this for the input in the model
    # returns image tensor (CxHxW)
    return img


def get_image(file):
    path = os.path.join(Constants.dir_individual_image, file)
    img = np.array(PILImage.open(path))
    t_img = transform_img(img)
    return t_img


if __name__ == "__main__":
    image_ = get_image("cats_and_dogs.jpg")
    epoch = Hyper.total_epochs
    model = load_checkpoint(epoch)
    individual_image(model, image_)