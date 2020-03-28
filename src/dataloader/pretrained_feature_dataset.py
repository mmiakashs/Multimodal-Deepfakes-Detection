import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import config

class DeepFakePretrainedDataset(Dataset):

    def __init__(self, data_dir_base_path,
                 modalities, dataset_type='train',
                 metadata_filename='metadata.json',
                 is_fake=False,
                 is_guiding=False):

        self.data_dir_base_path = data_dir_base_path
        self.dataset_type = dataset_type
        self.modalities = modalities
        self.metadata_filename = metadata_filename
        self.is_fake = is_fake
        self.is_guiding = is_guiding

        self.load_data()
        self.label_names = self.data.label.unique()
        self.num_labels, self.label_name_id, self.label_id_name = self.get_label_name_id(self.label_names)

    def load_data(self):
        self.data = pd.read_csv(self.data_dir_base_path+'/'+self.metadata_filename)
        self.data = self.data[self.data[config.dataset_split_tag] == self.dataset_type]
        if(self.is_fake):
            self.data = self.data[self.data['label'] == config.fake_label_tag]
        for i, row in self.data.iterrows():
            tm_filename = row[config.filename_tag]
            if(not os.path.exists(f'/data/research_data/dfdc_train_data/{tm_filename}')):
                self.data.at[i,'label'] = 'MISSING'
                
            tm_filename = row[config.original_filename_tag]
            if((row['label']==config.fake_label_tag) and (not os.path.exists(f'/data/research_data/dfdc_train_data/{tm_filename}'))):
                self.data.at[i,'label'] = 'MISSING'
        if(self.is_fake):
            self.data = self.data[self.data['label'] == config.fake_label_tag]
        
        self.data = self.data[self.data['label'] != 'MISSING']
            
        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def get_embedding(self, idx, filename_tag):
        filename = f'{self.data.loc[idx, filename_tag].split(".")[0]}.pt'
        data_filepath = f'{self.data_dir_base_path}/{filename}'
        seq = torch.load(data_filepath)
        seq = seq.detach()

        return seq, seq.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_label = self.data.loc[idx, config.label_tag]
        data = {}
        modality_mask = []
        
        # print(f'************ Start Data Loader for {idx} ************')
        if(self.is_guiding):
            modality = config.fake_modality_tag
            seq, seq_len = self.get_embedding(idx, config.filename_tag)
            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len
            modality_mask.append(True if seq_len == 0 else False)
        else:
            if(data_label==config.real_label_tag):
                modality = config.real_modality_tag
                seq, seq_len = self.get_embedding(idx, config.filename_tag)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)

                modality = config.fake_modality_tag
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)

            else:
                modality = config.real_modality_tag
                seq, seq_len = self.get_embedding(idx, config.original_filename_tag)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)

                modality = config.fake_modality_tag
                seq, seq_len = self.get_embedding(idx, config.filename_tag)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)

        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()
        data[config.label_tag] = self.label_name_id[str(data_label)]
        data[config.modality_mask_tag] = modality_mask

        # print(f'************ End Data Loader for {idx} ************')
        return data

    def get_label_name_id(self, label_names):

        label_names = sorted(label_names)
        num_labels = len(label_names)

        temp_dict_type_id = {label_names[i]: i for i in range(len(label_names))}
        temp_dict_id_type = { i : label_names[i] for i in range(len(label_names))}
        return num_labels, temp_dict_type_id, temp_dict_id_type

modalities = [config.fake_modality_tag]

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
        
#         print(f'{modality} seq lengths: ',data[modality + config.modality_seq_len_tag])

        seq_max_len = data[modality + config.modality_seq_len_tag].max()
        seq_mask = torch.stack(
            [gen_mask(seq_len, seq_max_len)  for seq_len in data[modality + config.modality_seq_len_tag]], dim=0)
        data[modality + config.modality_mask_suffix_tag] = seq_mask

    data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                dtype=torch.long)
    data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
    return data

