from sklearn.preprocessing import LabelBinarizer
from typing import Union, Optional
from torch.utils.data import Dataset
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import pandas as pd
import math
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image

path_img_dir = "C:/cache/torchok/data/sop/Stanford_Online_Products"
path_train = "C:/cache/torchok/data/sop/Stanford_Online_Products/Ebay_train.txt"


class SopDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # data loading
        self.train_data = pd.read_csv(path_train,
                                      delimiter=" ")[:1000]
        self.img_paths = self.train_data['path']
        self.img_labels = self.train_data['class_id']
        self.img_dir = path_img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_samples = self.img_labels.shape[0]
        self.img_dim = (256, 256)
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_paths.iloc[index])
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.img_dim)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[index]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.target_transform:
            label = self.target_transform(label)['label']
        return image, label

    def __len__(self):
        return self.n_samples


