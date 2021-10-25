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
from model import CustomEfficientNet
from util import *
from dataset import *
from loss import *
from transforms import *


import warnings 
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOGGER = init_logger()
seed_torch(seed=CFG.seed)

test = os.listdir('test')
test = pd.DataFrame({'BraTS21ID' : test})
test['BraTS21ID'] = test['BraTS21ID'].astype(int)

model_eff = CustomEfficientNet(CFG.model_name, pretrained=False)

states = [load_state_eff(f'{CFG.weights}/efficientnet_b3_fold0_best.pth'),
          load_state_eff(f'{CFG.weights}/efficientnet_b3_fold1_best.pth'),
          load_state_eff(f'{CFG.weights}/efficientnet_b3_fold2_best.pth'),
          load_state_eff(f'{CFG.weights}/efficientnet_b3_fold3_best.pth'),
          load_state_eff(f'{CFG.weights}/efficientnet_b3_fold4_best.pth'),
]

test_dataset = TestDataset(test, transform=get_transforms(data='valid'))

test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True)
predictions = inference(model_eff, states, test_loader, device)

# submission
test['MGMT_value'] = predictions[:,1]
test[['BraTS21ID', 'MGMT_value']].to_csv(OUTPUT_DIR+'submission.csv', index=False)

