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
import torch.nn as nn

from src.utils import config
from src.dataloader.deep_fake_dataset import *
import time

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

data_dir_base_path = '/data/research_data/dfdc_train_data'
embed_dir_base_path = '/data/research_data/dfdc_embed'

rgb_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

original_model = models.resnet50(pretrained=True)
feature_extractor = ResNet50Bottom(original_model)
# print(feature_extractor)
total_parsing = 0
start_time = time.time()
for dirname, _, filenames in os.walk(data_dir_base_path):
    for filename in filenames:
#         print('filename',filename)
        video = Video(f'{data_dir_base_path}/{filename}', transforms=rgb_transforms)
        seq, seq_len = video.get_all_frames()
        embed = feature_extractor(seq)
        filename = filename.split('.')[0]
        torch.save(embed, f'{embed_dir_base_path}/{filename}.pt')
#         print('embedding is saved to: ',f'{embed_dir_base_path}/{filename}.pt')
#         print('embedding size', embed.size())
#         break
        total_parsing += 1
        if(total_parsing%1000==0):
            print(f'parsing completed:{total_parsing}')

print("--- %s seconds ---" % (time.time() - start_time))
print(f'total file parsing {total_parsing}')
print('feature_extraction_complete')