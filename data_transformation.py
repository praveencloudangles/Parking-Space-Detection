from torch.utils.data import Dataset
import os
import torch
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import dill as pickle
import glob

class CustomDataset(Dataset):
    def __init__(self, img_path, annot_path, width, height, transforms=None):
        self.transforms = transforms
        self.img_path = img_path
        self.annot_path = annot_path
        self.height = height
        self.width = width
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.img_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

        # Dynamically extract classes from the dataset
        self.classes = self.extract_classes()
        print("self.classes------------------",self.classes)

    def extract_classes(self):
        classes_set = set()
        for image_name in self.all_images:
            annot_filename = os.path.splitext(image_name)[0] + '.xml'
            annot_file_path = os.path.join(self.annot_path, annot_filename)

            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            for member in root.findall('object'):
                classes_set.add(member.find('name').text)

        # Convert set to a sorted list
        classes_list = sorted(list(classes_set))
        return classes_list

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)
        image = cv2.imread(image_path)
        print("image ---------------------------- :",image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        

        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.annot_path, annot_filename)

        boxes = []
        labels = []
        tree = ET.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            xmin = float(member.find('xmin').text)
            xmax = float(member.find('xmax').text)
            ymin = float(member.find('ymin').text)
            ymax = float(member.find('ymax').text)

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            if xmax_final == xmin_final:
                xmin_final -= 1
            if ymax_final == ymin_final:
                ymin_final -= 1
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = {
                'image': image_resized,
                'boxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(sample)
            image_resized = sample['image']
            target['boxes'] = sample['boxes']


        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)       

#-----------------------------------------------------------------------------------------
# Define your transformations here
class ToTensor(object):
    def __call__(self, sample):
        print("sample-------------------",sample)
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        image = image.transpose((2, 0, 1))  # Swap color axis
        return {'image': torch.tensor(image, dtype=torch.float32),
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels)}


# Define your paths for training and validation datasets
train_image_dir = 'datasets/train/images/'
train_annotation_dir = 'datasets/train/annotations/'
valid_image_dir = 'datasets/val/images/'
valid_annotation_dir = 'datasets/val/annotations/'

RESIZE_TO = 640
# Create datasets
train_dataset = CustomDataset(train_image_dir, train_annotation_dir, transforms=ToTensor(), width = RESIZE_TO, height = RESIZE_TO)
print("one-------------",train_dataset)
valid_dataset = CustomDataset(valid_image_dir, valid_annotation_dir, transforms=ToTensor(), width = RESIZE_TO, height = RESIZE_TO)
print("-------------",valid_dataset)

i, a = train_dataset[0]
print("iiiiii:",i)
print("aaaaa:",a)


with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open('valid_dataset.pkl', 'wb') as f:
    pickle.dump(valid_dataset, f)