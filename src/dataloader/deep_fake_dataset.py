import os
import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from src.utils import config

data_dir_base_path = '/data/research_data/dfdc_sample_data/train_sample_videos'


#imageio read frame as (channel, height, width)
class Video:
    def __init__(self, path, seq_max_len=None, transforms=None):
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
            if (self.transforms != None):
                frame = self.transforms(frame)
            frames.append(frame)

        seq = torch.stack(frames, dim=0).float()
        seq_len = seq.size(0)
        self.container.close()
        return seq, seq_len

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


class DeepFakeDataset(Dataset):

    def __init__(self, base_dir,
                 modalities, dataset_type='train',
                 window_size=1, window_stride=1,
                 seq_max_len=300, transforms_modalities=None,
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
        self.data = pd.read_json(self.base_dir+'/'+self.metadata_filename, orient='index')
        self.data = self.data[self.data[config.dataset_split_tag] == self.dataset_type]
        if(self.restricted_labels!=None):
            for restricted_label in self.restricted_labels:
                for restricted_id in self.restricted_ids:
                    self.data = self.data[self.data[restricted_label]!=restricted_id]

        self.data.index.name = 'filename'
        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def get_video_data(self, idx, modality, filename_tag):
        data_filepath = f'{self.base_dir}/{self.data.loc[idx, filename_tag]}'
        
        if (not pd.isna(self.data.loc[idx, filename_tag]) and os.path.exists(data_filepath)):
            video = Video(data_filepath, self.seq_max_len, self.transforms_modalities[modality])
            seq, seq_len = video.get_all_frames()
        else:
            seq = np.zeros((self.seq_max_len, config.image_channels,
                                 config.image_width, config.image_height))
            seq = torch.from_numpy(seq).float()
            seq_len = 0

        return seq, seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_label = [ self.data.loc[idx,[config.label_tag]] ]
        data_label = str(data_label[0])

        data = {}
        modality_mask = []
        
        # print(f'************ Start Data Loader for {idx} ************')
        if(data_label==config.real_label_tag):
            modality = f'{config.original_modality_tag}_{config.rgb_modality_tag}'
            seq, seq_len = self.get_video_data(idx, config.rgb_modality_tag, config.original_filename_tag)
            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len
            modality_mask.append(True if seq_len == 0 else False)

        else:
            modality = f'{config.original_modality_tag}_{config.rgb_modality_tag}'
            seq, seq_len = self.get_video_data(idx, config.rgb_modality_tag, config.original_filename_tag)
            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len
            modality_mask.append(True if seq_len == 0 else False)

            modality = f'{config.fake_modality_tag}_{config.rgb_modality_tag}'
            seq, seq_len = self.get_video_data(idx, config.rgb_modality_tag, config.filename_tag)
            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len
            modality_mask.append(True if seq_len == 0 else False)

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

modalities = [f'{config.fake_modality_tag}_{config.rgb_modality_tag}',
              f'{config.original_modality_tag}_{config.rgb_modality_tag}']

def gen_mask(seq_len, max_len):
   return torch.arange(max_len) > seq_len

def pad_collate(batch):
   batch_size = len(batch)
   data = {}
   for modality in modalities:
       data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
       data[modality + config.modality_seq_len_tag] = torch.tensor(
               [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
               dtype=torch.float)

       seq_max_len = data[modality + config.modality_seq_len_tag].max()
       seq_mask = torch.stack(
           [gen_mask(seq_len, seq_max_len) for seq_len in data[modality + config.modality_seq_len_tag]],
           dim=0)
       data[modality + config.modality_mask_tag] = seq_mask

   data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                dtype=torch.long)
   data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
   return data
