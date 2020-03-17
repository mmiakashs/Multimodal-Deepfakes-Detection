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


class Video:
    def __init__(self, path, seq_max_len, transforms):
        self.path = path
        self.seq_max_len = seq_max_len
        self.transforms = transforms
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']

    def get_all_frames(self):
        self.init_head()
        frames = []
        for idx, frame in enumerate(self.container):
            frame = Image.fromarray(frame)
            if(self.transforms!=None):
                frame = self.transforms(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        mask = self.gen_mask(frames.shape[0], self.seq_max_len).bool()
        frames = self.pad_seq(frames).float()
        return frames, mask
    
    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def pad_seq(self, seq):
        seq = seq[:self.seq_max_len, :]
        seq_len = seq.shape[0]
        padding = np.zeros((self.seq_max_len - seq_len, seq.shape[1], seq.shape[2], seq.shape[3]))
        seq = np.concatenate((seq, padding), axis=0)
        seq = torch.from_numpy(seq).float()
        return seq

    def init_head(self):
        self.container.set_image_index(0)

    def next_frame(self):
        self.container.get_next_data()

    def get(self, key):
        return self.container.get_data(key)

    def __call__(self, key):
        return self.get(key)

    def __len__(self):
        return self.length

#Dataset for NTU-RGBD-120
class DeepFakeDataset(Dataset):

    def __init__(self, base_dir,
                 modalities, dataset_type='train',
                 window_size=1, window_stride=1,
                 seq_max_len=200, transforms_modalities=None,
                 restricted_ids=None, restricted_labels=None,
                 metadata_filename='metadata.json'):

        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.transforms_modalities = transforms_modalities
        self.restricted_ids = restricted_ids
        self.restricted_labels = restricted_labels

        self.modalities = modalities
        self.seq_max_len = seq_max_len
        self.metadata_filename = metadata_filename

        self.window_size = window_size
        if (window_stride == None):
            self.window_stride = window_size
        else:
            self.window_stride = window_stride

        self.load_data()
        self.label_names = self.data.label.unique()
        self.num_labels, self.label_name_id, self.label_id_name = self.get_label_name_id(self.label_names)

    def load_data(self):
        if(self.dataset_type==config.train_dataset_tag):
            self.data = pd.read_json(self.base_dir+'/'+self.metadata_filename, orient='index')
            self.data = self.data[self.data[config.dataset_split_tag] == self.dataset_type]
            if(self.restricted_labels!=None):
                for restricted_label in self.restricted_labels:
                    for restricted_id in self.restricted_ids:
                        self.data = self.data[self.data[restricted_label]!=restricted_id]

            self.data.index.name = 'fake'
            self.data.reset_index(inplace=True)

        elif(self.dataset_type==config.test_dataset_tag):
            pass

    def __len__(self):
        return len(self.data)

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def get_video_data(self, path, modality):
        video = Video(path, self.seq_max_len, self.transforms_modalities[modality])
        frames, mask = video.get_all_frames()
        return frames, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_label = [self.data.loc[idx,[config.label_tag]]]
        data_label = [self.label_name_id[str(temp[0])] for temp in data_label]

        data = {}
        modality_mask = []
        # print(f'************ Start Data Loader for {idx} ************')
        for modality in self.modalities:

            tm_modality = modality
            if(self.label_id_name[data_label[0]]==config.real_label_tag and modality==config.original_modality):
                tm_modality = config.fake_modality
            # print(self.label_id_name[data_label[0]], modality, tm_modality)

            data_filepath = f'{self.base_dir}/{self.data.loc[idx, tm_modality]}'
            # print(data_filepath)
            # print(pd.isna(self.data.loc[idx, tm_modality]), os.path.exists(data_filepath))
            tm_seq_len = 0
            if (not pd.isna(self.data.loc[idx, tm_modality]) and os.path.exists(data_filepath)):
                temp_seq, temp_seq_mask = self.get_video_data(data_filepath, tm_modality)
                tm_seq_len = temp_seq.shape[0]
            else:
                temp_seq = np.zeros((self.seq_max_len, config.image_channels,
                                     config.image_width, config.image_height))
                temp_seq = torch.from_numpy(temp_seq).float()
                temp_seq_mask = self.gen_mask(0, self.seq_max_len)

            data[modality + config.modality_mask_suffix_tag] = temp_seq_mask
            data[modality] = temp_seq
            # print(idx, tm_modality, tm_seq_len)
            
            if (tm_seq_len == 0):
                modality_mask.append(True)
            else:
                modality_mask.append(False)

        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()

        data[config.label_tag] = data_label[0]
        data[config.modality_mask_tag] = modality_mask

        # print(f'************ End Data Loader for {idx} ************')
        return data

    def get_label_name_id(self, label_names):

        label_names = sorted(label_names)
        num_labels = len(label_names) + 1

        temp_dict_type_id = {label_names[i]: i + 1 for i in range(len(label_names))}
        temp_dict_id_type = { i+1 : label_names[i] for i in range(len(label_names))}
        return num_labels, temp_dict_type_id, temp_dict_id_type


def get_ids_from_split(split_ids, split_index, validation_type='trial'):
    temp = []
    for id in split_index:
        temp.append(split_ids[id])

    restricted_person_ids = None
    restricted_sample_ids = None

    if(validation_type=='trial'):
        restricted_sample_ids = temp
    else:
        restricted_person_ids = temp
    return restricted_sample_ids, restricted_person_ids

# Debugging
rgb_transforms = transforms.Compose([
                transforms.Resize((config.image_width, config.image_height)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])

# video = Video('../../../mm_har_datasets/ntu_rgb_d_120/data/rgb/S001C001P001R002A001_rgb.avi',
#               transforms=rgb_transforms)
# frames = video.get_all_frames()
# print(frames.size())
# print(frames[:,1,:,:].size())

# Debugging

modalities = [config.fake_modality, config.original_modality]
transforms_modalities = {config.fake_modality: rgb_transforms,
                         config.original_modality: rgb_transforms}

train_dataset = DeepFakeDataset(base_dir='../../../deep_fake_sample_dataset/data/sample/train_sample_videos',
                                modalities=modalities,
                                dataset_type=config.train_dataset_tag, transforms_modalities=transforms_modalities,
                                seq_max_len=200, window_size=1, window_stride=1)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
for  batch_id, batch in enumerate(train_dataloader, 0):
    print(batch[config.label_tag])
    for modality in modalities:
        print(f'{modality} size:',batch[modality].size())
        print(f'{modality} mask size:', batch[modality+'_mask'].size())
    print(f'modality mask size:', batch['modality_mask'].size())
    print(f'label size:', batch['label'].size())
    print(batch["modality_mask"])
