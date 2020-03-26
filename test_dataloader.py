import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.dataloader.pretrained_feature_dataset import DeepFakePretrainedDataset
from src.utils import config
from src.dataloader.deep_fake_dataset import *

# data_dir_base_path = '/data/research_data/dfdc_sample_data/train_sample_videos'
data_dir_base_path = '/data/research_data/dfdc_train_data'

# Debugging
rgb_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])

video = Video(f'{data_dir_base_path}/aagfhgtpmv.mp4', transforms=rgb_transforms)
print(video.get_all_frames()[0].size())

modalities = [f'{config.real_modality_tag}',
              f'{config.fake_modality_tag}']

transforms_modalities = {modalities[0]: rgb_transforms,
                         modalities[1]: rgb_transforms}

# train_dataset = DeepFakeDataset(data_dir_base_path=data_dir_base_path,
#                                 modalities=modalities,
#                                 dataset_type=config.train_dataset_tag, transforms_modalities=transforms_modalities,
#                                 seq_max_len=300, window_size=5, window_stride=5,
#                                 metadata_filename='metadata.csv',
#                                 is_fake=True)
train_dataset = DeepFakePretrainedDataset(data_dir_base_path='/data/research_data/dfdc_embed_small',
                                   modalities=modalities,
                                   dataset_type='train',
                                   metadata_filename='metadata.csv')

train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=pad_collate, num_workers=2)
for  batch_id, batch in enumerate(train_dataloader, 0):
    print(batch[config.label_tag])
    for modality in modalities:
        if(batch[modality] is not None):
            print(f'{modality} size:',batch[modality].size())
            print(f'{modality} mask size:', batch[modality+'_mask'].size())
        else:
            print(f'{modality} size: None')
            print(f'{modality} mask size: 0')
            
    print(f'modality mask size:', batch['modality_mask'].size())
    print(f'label size:', batch['label'].size())
    print(batch["modality_mask"])