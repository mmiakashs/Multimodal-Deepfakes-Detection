import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import config
from .Vis_Module import Vis_Module


class DeepFakeTSModel(nn.Module):
    def __init__(self, mm_module_properties,
                 modalities,
                 window_size, window_stride,
                 modality_embedding_size,
                 module_networks,
                 batch_first=True,
                 multi_modal_nhead=4,
                 mm_embedding_attn_merge_type='sum',
                 dropout=0.1,
                 activation="relu",
                 is_guiding=False,
                 num_activity_types=2):
        super(DeepFakeTSModel, self).__init__()

        self.mm_module_properties = mm_module_properties
        self.modalities = modalities
        self.num_modality = len(modalities)
        self.batch_first = batch_first

        self.multi_modal_nhead = multi_modal_nhead
        self.mm_embedding_attn_merge_type = mm_embedding_attn_merge_type
        self.dropout = dropout
        self.activation = activation
        self.window_size = window_size
        self.window_stride = window_stride
        self.lstm_bidirectional = False
        self.modality_embedding_size = modality_embedding_size
        self.module_networks = module_networks
        self.num_module_networks = len(self.module_networks)
        self.is_guiding = is_guiding
        self.num_activity_types = 2

        print('module_networks', self.module_networks)
        self.mm_module = nn.ModuleDict()
        for modality in self.module_networks:
            self.mm_module[modality] = Vis_Module(cnn_in_channel=self.mm_module_properties[modality]['cnn_in_channel'],
                                                  feature_embed_size=self.mm_module_properties[modality]['feature_embed_size'],
                                                  kernel_size=self.mm_module_properties[modality]['kernel_size'],
                                                  lstm_hidden_size=self.mm_module_properties[modality]['lstm_hidden_size'],
                                                  fine_tune=self.mm_module_properties[modality]['fine_tune'],
                                                  batch_first=self.batch_first,
                                                  window_size=self.window_size,
                                                  window_stride=self.window_stride,
                                                  n_head=self.mm_module_properties[modality]['module_embedding_nhead'],
                                                  dropout=self.mm_module_properties[modality]['dropout'],
                                                  activation=self.mm_module_properties[modality]['activation'],
                                                  encoder_num_layers=self.mm_module_properties[modality]['lstm_encoder_num_layers'],
                                                  lstm_bidirectional=self.mm_module_properties[modality]['lstm_bidirectional'],
                                                  lstm_dropout=self.mm_module_properties[modality]['lstm_dropout'],
                                                  pool_fe_kernel=self.mm_module_properties[modality]['feature_pooling_kernel'],
                                                  pool_fe_stride=self.mm_module_properties[modality]['feature_pooling_stride'],
                                                  pool_fe_type=self.mm_module_properties[modality]['feature_pooling_type'],
                                                  is_guiding=self.is_guiding)

            if (self.mm_module_properties[modality]['lstm_bidirectional']):
                self.lstm_bidirectional = True

        if (self.lstm_bidirectional):
            self.modality_embedding_size = 2 * self.modality_embedding_size

        self.mm_embeddings_bn = nn.BatchNorm1d(self.num_module_networks)
        self.mm_embeddings_relu = nn.ReLU()
        self.mm_embeddings_dropout = nn.Dropout(p=self.dropout)

        self.mm_mhattn = nn.MultiheadAttention(embed_dim=self.modality_embedding_size,
                                               num_heads=self.multi_modal_nhead,
                                               dropout=self.dropout)

        self.mm_mhattn_bn = nn.BatchNorm1d(self.num_module_networks)
        self.mm_mhattn_relu = nn.ReLU()
        self.mm_mhattn_dropout = nn.Dropout(p=self.dropout)

        if (self.mm_embedding_attn_merge_type == 'sum'):
            if (self.lstm_bidirectional):
                self.fc_output1 = nn.Linear(self.num_module_networks * self.modality_embedding_size,
                                            self.num_module_networks * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 2,
                                            self.num_module_networks * self.modality_embedding_size // 4)
                self.fc_output3 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 4,
                                            self.num_module_networks * self.num_activity_types)
            else:
                self.fc_output1 = nn.Linear(self.num_module_networks * self.modality_embedding_size,
                                            self.num_module_networks * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 2,
                                            self.num_module_networks * self.num_activity_types)
        else:
            if (self.lstm_bidirectional):
                self.fc_output1 = nn.Linear(self.num_module_networks * self.modality_embedding_size,
                                            self.num_module_networks * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 2,
                                            self.num_module_networks * self.modality_embedding_size // 4)
                self.fc_output3 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 4,
                                            self.num_activity_types)
            else:
                self.fc_output1 = nn.Linear(self.num_module_networks * self.modality_embedding_size,
                                            self.num_module_networks * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_module_networks * self.modality_embedding_size // 2,
                                            self.num_activity_types)

        self.module_attn_weights = None
        self.mm_attn_weight = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)

        nn.init.xavier_uniform_(self.fc_output2.weight)
        nn.init.constant_(self.fc_output2.bias, 0.)

        if (self.lstm_bidirectional):
            nn.init.xavier_uniform_(self.fc_output3.weight)
            nn.init.constant_(self.fc_output3.bias, 0.)

    def forward(self, input):
        # print('########### Start MM_HAR_Module ###########')
        attn_output = {}
        self.module_attn_weights = {}
        for module_network in self.module_networks:
            tm_attn_output, self.module_attn_weights[module_network] = \
                self.mm_module[module_network](input[config.fake_modality_tag],
                                               input[config.real_modality_tag],
                                               input[config.fake_modality_tag+config.modality_mask_suffix_tag],
                                               input[config.real_modality_tag + config.modality_mask_suffix_tag])
#             tm_attn_output = torch.sum(tm_attn_output, dim=1).squeeze(dim=1)
            attn_output[module_network] = tm_attn_output
                # print(f'attn_output[{modality}] size: {attn_output[modality].size()}')

        mm_embeddings = []
        for modality in self.module_networks:
#             print(f'{modality} embedding size {attn_output[modality].size()}')
            mm_embeddings.append(attn_output[modality])

        mm_embeddings = torch.stack(mm_embeddings, dim=1).contiguous()
        # mm_embeddings = self.mm_embeddings_dropout(self.mm_embeddings_relu(self.mm_embeddings_bn(mm_embeddings)))
        mm_embeddings = self.mm_embeddings_relu(self.mm_embeddings_bn(mm_embeddings))

        nbatches = mm_embeddings.shape[0]
        # transpose batch and sequence (B x S x ..) --> (S x B x ..)
        mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()
        mattn_output, self.mm_attn_weight = self.mm_mhattn(mm_embeddings, mm_embeddings, mm_embeddings)
        mattn_output = mattn_output.transpose(0,1).contiguous()  # transpose batch and sequence (S x B x ..) --> (B x S x ..)

        # print('mattn_output',mattn_output.size())
        mattn_output = self.mm_mhattn_dropout(self.mm_mhattn_relu(self.mm_mhattn_bn(mattn_output)))

        if(self.mm_embedding_attn_merge_type=='sum'):
            mattn_output = torch.sum(mattn_output, dim=1).squeeze(dim=1)

        mattn_output = mattn_output.contiguous().view(nbatches, -1)

        # print('mattn_output shape', mattn_output.size())
        if (self.lstm_bidirectional):
            output = self.fc_output1(mattn_output)
            output = self.fc_output2(output)
            output = self.fc_output3(output)
        else:
            output = self.fc_output1(mattn_output)
            output = self.fc_output2(output)

        # print('final output shape', output.size())
        # print('########### End MM_HAR_Module ###########')

        return F.log_softmax(output, dim=1)
