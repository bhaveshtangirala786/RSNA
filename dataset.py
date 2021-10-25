import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm
from cfg import CFG

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['BraTS21ID'].values
        self.labels = df['MGMT_value'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        root = f'{CFG.TRAIN_PATH}/{str(self.file_names[idx]).zfill(5)}/'
        com = []
        for typ in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
            paths = os.listdir(root + typ)
            rnd = random.sample(paths, min(10,len(paths)))
            typ_imgs = []
            for f in rnd:
                file_path = f'{root}{typ}/{f}'
                image = cv2.imread(file_path)[:,:,0]
                typ_imgs.append(cv2.resize(image, (CFG.size, CFG.size)))
            com.append(np.mean(typ_imgs, axis = 0))
        image = np.array(com).transpose((1,2,0)) / 255
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            image = image.float()
        label = torch.tensor(self.labels[idx]).long()
        return image, label


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['BraTS21ID'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        root = f'{CFG.TEST_PATH}/{str(self.file_names[idx]).zfill(5)}/'
        com = []
        for typ in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
            paths = os.listdir(root + typ)
            rnd = random.sample(paths, min(10, len(paths)))
            typ_imgs = []
            for f in rnd:
                file_path = f'{root}{typ}/{f}'
                if CFG.test_type == 'dcm':
                    dicom = pydicom.read_file(file_path)
                    data = apply_voi_lut(dicom.pixel_array, dicom)
                    if dicom.PhotometricInterpretation == "MONOCHROME1":
                        data = np.amax(data) - data
                    data = data - np.min(data)
                    data = data / np.max(data)
                    image = (data * 255).astype(np.uint8)
                else:
                    image = cv2.imread(file_path)[:,:,0]
                typ_imgs.append(cv2.resize(image, (CFG.size, CFG.size)))
            com.append(np.mean(typ_imgs, axis = 0))
        image = np.array(com).transpose((1,2,0)) / 255
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            image = image.float()
        return image