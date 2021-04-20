import os, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data
from PIL import Image as PILImage
from torchvision import transforms as transforms
from config import Hyper, Constants

from xml.dom import minidom


# this class inherit Pytorch Dataset class
# loads 1 data point:
# 1 image and the vector of labels

class PascalVOC2012Dataset(data.Dataset):

    def __init__(self, **kwargs):
        # Classes from Pascal VOC 2012 dataset, in the correct order without the bgr
        self.voc_classes = kwargs['classes']
        self.dir = kwargs['dir']
        self.dir_bbox = kwargs['dir_label_bbox']
        self.img_max_size = kwargs['img_max_size']
        self.imgs = os.listdir(self.dir)
        self._classes = kwargs['classes']

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    # Convert into Pytorch Tensor and transform back to large values by multiplying by 255
    def transform_img(self, img, img_max_size):
        h, w, c = img.shape
        h_, w_ = img_max_size[0], img_max_size[1]
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
        # need this for the input in the model
        # returns image tensor (CxHxW)
        return img

    # load one image
    # idx: index in the list of images
    def load_img(self, idx):
        # im = cv2.imread(os.path.join(self.dir, self.imgs[idx]))
        # im = self.transform_img(im, self.img_max_size)
        im = np.array(PILImage.open(os.path.join(self.dir, self.imgs[idx])))
        im = self.transform_img(im, self.img_max_size)
        return im

    # this method returns the size of the object inside the bounding box:
    # input is a list in format xmin,ymin, xmax,ymax
    def get_size(self, bbox):
        _h, _w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        size = _h * _w
        return size

    # compute IoU overlap between given bboxes
    # i mage size: HxW
    def get_iou(self, bbox1, bbox2, img_size):
        s1 = self.get_size(bbox1)
        s2 = self.get_size(bbox2)
        mask = np.zeros(img_size)
        mask[int(bbox1[0]):int(bbox1[2]), int(bbox1[1]):int(bbox1[3])] = 1
        mask[int(bbox2[0]):int(bbox2[2]), int(bbox2[1]):int(bbox2[3])] += 1
        #  if bboxes intersect, there's an area with values > 1
        intersect = np.sum(mask > 1)
        if intersect > 0:
            iou = intersect / (s1 + s2 - intersect + 1)
        else:
            iou = 0  # false positive
        return iou

    # fname must be an XML file with the name of the class of the object in the bbox,
    # bbox coordinates (xmin, xmax, ymin, ymax) 
    # and a class name 
    # this method returns the mask too
    def extract_bboxes_and_masks_pascal(self, idx, list_of_classes):
        fname = self.imgs[idx].split('.')[0] + '.xml'
        fname = os.path.join(self.dir_bbox, fname)

        classes = []
        bboxes = []
        pascal_voc_xml_doc = minidom.parse(fname)
        # This returns the list of objects tagged 'name'
        nameval = pascal_voc_xml_doc.getElementsByTagName('name')
        # get the number of total objects in the image
        total_objects = len(nameval)
        wrong_class_ids = []
        good_class_id = []
        name_list = list_of_classes.keys()
        for id, n in enumerate(nameval):
            name = n.firstChild.nodeValue
            # some classes not in Pascal VOC data set (e.g.head)
            if name not in name_list:
                total_objects -= 1
                wrong_class_ids.append(id)
            else:
                good_class_id.append(id)
                class_id = list_of_classes[name]
                classes.append(class_id)

        xminvals = []
        yminvals = []
        xmaxvals = []
        ymaxvals = []
        # this returnx xmin, ymin, xmax and ymax values of the bbox
        xminval = pascal_voc_xml_doc.getElementsByTagName('xmin')
        yminval = pascal_voc_xml_doc.getElementsByTagName('ymin')
        xmaxval = pascal_voc_xml_doc.getElementsByTagName('xmax')
        ymaxval = pascal_voc_xml_doc.getElementsByTagName('ymax')
        # get xmins for all objects
        for idz, xmincode in enumerate(xminval):
            xmin = float(xmincode.firstChild.nodeValue)
            if idz in good_class_id:
                xminvals.append(xmin)

        # get ymins for all objects
        for idz, ymincode in enumerate(yminval):
            ymin = float(ymincode.firstChild.nodeValue)
            if idz in good_class_id:
                yminvals.append(ymin)

        # get xmaxs for all objects
        for idz, xmaxcode in enumerate(xmaxval):
            xmax = float(xmaxcode.firstChild.nodeValue)
            if idz in good_class_id:
                xmaxvals.append(xmax)

        # get ymaxs for all objects
        for idz, ymaxcode in enumerate(ymaxval):
            ymax = float(ymaxcode.firstChild.nodeValue)
            if idz in good_class_id:
                ymaxvals.append(ymax)

        # put the coordinates together
        for i in range(total_objects):
            obj = [xminvals[i], yminvals[i], xmaxvals[i], ymaxvals[i]]
            bboxes.append(obj)
        # output a dictionary with classes and bboxes
        label = {}
        label['labels'] = torch.tensor(classes, dtype=torch.long)
        label['boxes'] = torch.as_tensor(bboxes)
        return label

    [staticmethod]
    def get_data_loader(dir):
        pascal_voc_classes = {}
        for id, name in enumerate(Hyper.pascal_categories):
            pascal_voc_classes[name] = id
        instance_data_args = {'classes': pascal_voc_classes,
                              'img_max_size': Hyper.img_max_size,
                              'dir': dir,
                              'dir_label_bbox': Constants.dir_label_bbox}
        instance_data_point = PascalVOC2012Dataset(**instance_data_args)
        instance_dataloader_args = {'batch_size': Hyper.batch_size}
        instance_dataloader = data.DataLoader(instance_data_point, **instance_dataloader_args)
        return instance_dataloader

    def __len__(self):
        return len(os.listdir(self.dir))

        # iterates through the dataset directory to return the image and its gt

    def __getitem__(self, idx):
        # here you have to implement functionality using the methods in this class to return X (image) and y (its label)
        # X must be dimensionality (3,max_size[1], max_size[0]) if you use VGG16
        # y must be dimensioanlity (self.voc_classes)
        X = self.load_img(idx)
        y = self.extract_bboxes_and_masks_pascal(idx, self._classes)
        return idx, X, y
