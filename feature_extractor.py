import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.utils import config
from src.dataloader.deep_fake_dataset import *

data_dir_base_path = '/data/research_data/dfdc_train_data'
rgb_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

for dirname, _, filenames in os.walk(data_dir_base_path):
    for filename in filenames:
        video = Video(f'{data_dir_base_path}/aagfhgtpmv.mp4', transforms=rgb_transforms)
        frames = video.get_all_frames()[0].size()