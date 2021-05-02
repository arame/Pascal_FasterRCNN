from config import Hyper, Constants, OutputStore
from utils import load_checkpoint
import os
import cv2
from PIL import Image as PILImage
import lime
from lime import lime_image
import numpy as np
import torch
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries 
from skimage import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_lime_predictions(fasterrcnn_model, img, path): 
    iter = 3500
    top_labels = 5
    num_super_pixels = 10
    fasterrcnn_model.eval()
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img, fasterrcnn_model, top_labels = top_labels, hide_color = 0, num_samples = iter)
    no_labels = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(no_labels, positive_only = False, num_features = num_super_pixels, hide_rest = False)
    lime_img = mark_boundaries(temp/2 + .5, mask)
    file_lime = path.replace(".jpg", "_lime.jpg")
    print(f"output file {file_lime}")
    cv2.imwrite(file_lime, lime_img)


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
    p_img = PILImage.open(path)
    img = np.array(p_img)
    t_img = transform_img(img)
    return t_img, img, path


def process_images(image_file):
    t_image_, img_, path_ = get_image(image_file)
    #epoch = Hyper.total_epochs
    epoch = 13
    model, _ = load_checkpoint(epoch)
    get_lime_predictions(model, img_, path_)
    individual_image(model, t_image_, img_, path_)


if __name__ == "__main__":
    OutputStore.check_folder(Constants.dir_individual_image)
    image_files = ["aeroplane_with_persons.jpg", "cats_and_dogs.jpg", "sofa_dog_person.jpg", "object_detection.jpg"]
    for image_file in image_files:
        process_images(image_file)



