import torch as T
import os

class Hyper:
    total_epochs = 50
    learning_rate = 1e-6
    batch_size = 1
    box_score_thresh = 0.75
    anchor_size = 256    # default is 256
    img_max_size = (512,512)
    pascal_categories = ['__bgr__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',  
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    num_classes = len(pascal_categories)
    selected_category = "person"
    pascal_problem = "instance"


    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"Number of epochs = {Hyper.total_epochs}")
        print(f"learning rate = {Hyper.learning_rate}")
        print(f"batch_size = {Hyper.batch_size}")


class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    selected_category = 'person'
    load_model = False
    save_model = True
    dir_images = "../pascal/train_data"
    dir_test_images = "../pascal/test_data"
    dir_val_images = "../pascal/val_data"
    dir_label_bbox = "../pascal/annotations"
    dir_individual_image = "images"



class OutputStore:
    dir_output_test_images = "../output_test"
    backup_pascal_model_folder = "../backup_pascal"
    backup_model_path = "../backup_pascal/model.pth"
    chart_path_pascal = "../charts_pascal"
    loss_filename = "losses_per_epoch.jpg"
    ave_MAP_filename = "ave_MAP_per_epoch.jpg"
    ave_IOU_filename = "ave_IOU_per_epoch.jpg"


    [staticmethod]
    def set_output_stores():
        OutputStore.check_folder(OutputStore.dir_output_test_images)
        OutputStore.check_folder(OutputStore.backup_pascal_model_folder)
        OutputStore.check_folder(OutputStore.chart_path_pascal)


    [staticmethod]
    def check_folder(folder):
        if os.path.isdir(folder) == False:
            os.mkdir(folder)


