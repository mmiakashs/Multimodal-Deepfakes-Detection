import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision import models

from src.utils import config
from src.dataloader.deep_fake_dataset import *

data_dir_base_path = '/data/research_data/dfdc_train_data'
embed_dir_base_path = '/data/research_data/dfdc_embed'

rgb_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

feature_extractor = models.resnet50(pretrained=True)
for dirname, _, filenames in os.walk(data_dir_base_path):
    for filename in filenames:
        video = Video(f'{data_dir_base_path}/{filename}', transforms=rgb_transforms)
        seq, seq_len = video.get_all_frames()
        embed = feature_extractor(seq)
        filename = filename.split('.')[0]
        torch.save(embed, f'{embed_dir_base_path}/{filename}.pt')
        print('embedding is saved to: ',f'{embed_dir_base_path}/{filename}.pt')
        print('embedding size', embed.size())
        break

print('feature_extraction_complete')