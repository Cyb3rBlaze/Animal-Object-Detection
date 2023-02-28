import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import numpy as np

import random


class Dataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir

        self.img_size = img_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize(img_size), # resize the images to 224x224 pixels
            transforms.ToTensor(), # convert the images to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # augmentations
            transforms.ColorJitter(0.1, 0.1),
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.05, 1))
        ])

        # need to change annotations for flip so seperate
        self.horizontal_flip = transforms.RandomHorizontalFlip(1.0)

        print("Loading dataset sample names...")

        self.images_list = []
        self.annotations_list = []

        for animal_dir in os.listdir(self.data_dir):
            for filename in os.listdir(os.path.join(self.data_dir, animal_dir)):
                if filename[len(filename)-3:] == "jpg":
                    self.images_list += [os.path.join(self.data_dir, animal_dir, filename)]
                    self.annotations_list += [os.path.join(self.data_dir, animal_dir, filename[:len(filename)-3] + "txt")]
        
        combined_list = list(zip(self.images_list, self.annotations_list))
        random.shuffle(combined_list)

        self.images_list, self.annotations_list = zip(*combined_list)

        print('Images: ' + str(len(self.images_list)))
        print('Annotations: ' + str(len(self.annotations_list)))


    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # processing image
        image = self.transform(Image.open(self.images_list[idx]))

        # x_center, y_center, w, h
        annotations = []

        # processing annotations
        with open(self.annotations_list[idx]) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                annotations += [[float(num) for num in line]]
        
        final_annotations = []

        # random flip augmentation
        if random.randint(1, 10) % 2 == 0:
            image = self.horizontal_flip(image)
            for annotation in annotations:
                final_annotations += [[annotation[0], 1-annotation[1], annotation[2], annotation[3], annotation[4]]]
        else:
            final_annotations = annotations

        final_targets = {"boxes": [], "labels": [], "area": [], "image_id": idx}
        
        # rescaling values to be compatible with yolov5
        for annotation in final_annotations:
            x_min = annotation[1] - annotation[3]/2
            y_min = annotation[2] - annotation[4]/2
            x_max = annotation[1] + annotation[3]/2
            y_max = annotation[2] + annotation[4]/2

            final_targets["boxes"] += [[x_min*self.img_size[0], y_min*self.img_size[1], x_max*self.img_size[0], y_max*self.img_size[1]]]
            final_targets["labels"] += [annotation[0] + 1]
            final_targets["area"] += [int((annotation[3]*self.img_size[0])*(annotation[4]*self.img_size[1]))]
        
        final_targets["boxes"] = torch.tensor(final_targets["boxes"], dtype=torch.float32).to(self.device)
        final_targets["labels"] = torch.tensor(final_targets["labels"], dtype=torch.int64).to(self.device)
        final_targets["area"] = torch.tensor(final_targets["area"], dtype=torch.int32).to(self.device)
        final_targets["image_id"] = torch.tensor(idx, dtype=torch.int64).to(self.device)

        return image.to(self.device), final_targets