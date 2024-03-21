import cv2
import numpy as np
from utils.util import parse_car_csv
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from custom_finetune_dataset import CustomFineTuneDataset
from custom_batch_sampler import CustomBatchSampler
from utils.util import check_dir

model = models.alexnet(pretrained=True)
print(model)
print()
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)
print(model)