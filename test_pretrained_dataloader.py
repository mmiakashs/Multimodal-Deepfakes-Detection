import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.dataloader.pretrained_feature_dataset import *
from src.utils import config

# data_dir_base_path = '/data/research_data/dfdc_sample_data/train_sample_videos'
embed_dir_base_path = '/data/research_data/dfdc_embed'

# Debugging
modalities = [f'{config.real_modality_tag}',
              f'{config.fake_modality_tag}']

train_dataset = DeepFakePretrainedDataset(data_dir_base_path=embed_dir_base_path,
                                modalities=modalities,
                                dataset_type=config.train_dataset_tag,
                                metadata_filename='metadata.csv',
                                is_fake=False)
print('length train dataset',len(train_dataset.data))
print('label_names', train_dataset.label_names)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
for batch_id, batch in enumerate(train_dataloader, 0):
    print('batch_id', batch_id)
#     if(batch_id<66486):
#         continue
    print(batch[config.label_tag])
    for modality in modalities:
        if(batch[modality] is not None):
            print(f'{modality} size:',batch[modality].size())
            print(f'{modality} mask size:', batch[modality+'_mask'].size())
        else:
            print(f'{modality} size: None')
            print(f'{modality} mask size: 0')
            
#     print(f'modality mask size:', batch['modality_mask'].size())
#     print(f'label size:', batch['label'].size())
#     print(batch["modality_mask"])