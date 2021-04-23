from config import Hyper, Constants, OutputStore
from utils import load_checkpoint
import os
import cv2
from PIL import Image as PILImage
import numpy as np
import torch
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

''' Code adapted from 
https://colab.research.google.com/drive/1eAUjzV3nZXkUXi0spPg6zUHtJWaUSzFk?usp=sharing#scrollTo=8Dbx7HZFIKhz '''
''' I got better results using the PILImage library to convert the image to a tensor.
    In order to annotate the image I had to use the cv2 library'''


def individual_image(fasterrcnn_model, t_image, img, path):
    fasterrcnn_model.eval()
    prediction = fasterrcnn_model(t_image)
    print(prediction)
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = torch.round(prediction[0]["scores"] * 10) / 10

    # this will help us create a different colour for each class
    COLOURS = np.random.uniform(0, 255, size=(Hyper.num_classes, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            cv2.putText(img, text, (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        lineType=cv2.LINE_AA)

        file_bb = path.replace(".jpg", f"_out{j}.jpg")
        print(f"output file {file_bb}")
        cv2.imwrite(file_bb, img)
    print("** Images saved, the end **")


# this is same transform used in the dataloader code.
def transform_img(img):
    t_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    img = t_(img)
    img = img.unsqueeze_(0)
    img = img.to(Constants.device)
    return img


def get_image(file):
    path = os.path.join(Constants.dir_individual_image, file)
    img = np.array(PILImage.open(path))
    t_img = transform_img(img)
    return t_img, img, path


def process_images(image_file):
    t_image_, img_, path_ = get_image(image_file)
    epoch = Hyper.total_epochs
    model = load_checkpoint(epoch)
    individual_image(model, t_image_, img_, path_)


if __name__ == "__main__":
    OutputStore.check_folder(Constants.dir_individual_image)
    image_files = ["aeroplane_with_persons.jpg", "cats_and_dogs.jpg", "sofa_dog_person.jpg"]
    for image_file in image_files:
        process_images(image_file)

