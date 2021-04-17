import torch as T

class Hyper:
    total_epochs = 20
    learning_rate = 1e-6
    batch_size = 1
    box_score_thresh = 0.75
    img_max_size = (512,512)
    pascal_categories = ['__bgr__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',  
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
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
    dir_label_bbox = "../pascal/annotations"
    backup_model_folder = "../backup_pascal"
    backup_model_path = "../backup_pascal/model.pth"


