import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Vis_Module(nn.Module):
    def __init__(self, feature_embed_size, lstm_hidden_size,
                 fine_tune=False, kernel_size=3, cnn_in_channel=3,
                 batch_first=True, window_size=5, window_stride=3, n_head=4,
                 dropout=0.1, activation="relu",
                 encoder_num_layers=2, lstm_bidirectional=False,
                 pool_fe_kernel=None, pool_fe_stride=None, pool_fe_type='max', lstm_dropout=0.1,
                 adaptive_pool_tar_squze_mul=None,
                 attention_type='mm', is_attention=False,
                 is_guiding=False):

        super(Vis_Module, self).__init__()

        self.cnn_in_channel = cnn_in_channel
        self.feature_embed_size = feature_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fine_tune = fine_tune
        self.batch_first = batch_first
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.activation = activation
        self.window_size = window_size
        self.window_stride = window_stride
        self.encoder_num_layers = encoder_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.pool_fe_kernel = pool_fe_kernel
        self.pool_fe_stride = pool_fe_stride
        self.pool_fe_type = pool_fe_type
        self.adaptive_pool_tar_squze_mul = adaptive_pool_tar_squze_mul
        self.attention_type = attention_type
        self.is_attention = is_attention
        self.is_guiding = is_guiding

        self.feature_extractor = models.resnet50(pretrained=True)
        if (self.fine_tune):
            self.set_parameter_requires_grad(self.feature_extractor, self.fine_tune)

        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, self.feature_embed_size)

        self.fe_relu = nn.ReLU()
        self.fe_dropout = nn.Dropout(p=self.dropout)
        if (self.pool_fe_kernel):
            if (self.pool_fe_type == 'max'):
                self.pool_fe = nn.MaxPool1d(kernel_size=self.pool_fe_kernel,
                                            stride=self.pool_fe_stride)
            else:
                self.pool_fe = nn.AvgPool1d(kernel_size=self.pool_fe_kernel,
                                            stride=self.pool_fe_stride)

        self.lstm = nn.LSTM(input_size=self.feature_embed_size,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=self.batch_first,
                            num_layers=self.encoder_num_layers,
                            bidirectional=self.lstm_bidirectional,
                            dropout=self.lstm_dropout)

        if (self.is_attention):
            if (self.lstm_bidirectional):
                self.self_attn = nn.MultiheadAttention(embed_dim=2 * self.feature_embed_size,
                                                       num_heads=n_head,
                                                       dropout=self.dropout)
            else:
                self.self_attn = nn.MultiheadAttention(embed_dim=self.feature_embed_size,
                                                       num_heads=n_head,
                                                       dropout=self.dropout)

        self.self_attn_weight = None
        self.module_fe_relu = nn.ReLU()
        self.module_fe_dropout = nn.Dropout(p=0)

    def set_parameter_requires_grad(self, model, fine_tune):
        for param in model.parameters():
            param.requires_grad = self.fine_tune

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len

    def forward(self, fake_input, real_input,
                fake_input_mask, real_input_mask,
                input_len):
        # print('########### Start Vis_Module ###########')
        # print('input shape', input.size())
        if (self.is_guiding):
            attn_output, self.self_attn_weight = self.fake_embed_guided_by_real(fake_input,
                                                                                real_input,
                                                                                fake_input_mask,
                                                                                real_input_mask)
        else:
            attn_output, self.self_attn_weight = self.fake_embed_self_guide(fake_input,
                                                                            fake_input_mask)
        # print('attn_output shape', attn_output.size())
        # print('########### End MM_Module ###########')
        return attn_output, self.self_attn_weight

    def fake_embed_guided_by_real(self, fake_input, real_input,
                                  fake_input_mask, real_input_mask):
        real_x = real_input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()
        fake_x = fake_input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()

        real_embed = self.feature_extractor(real_x).contiguous()
        fake_embed = self.feature_extractor(fake_x).contiguous()
        if (self.batch_first):
            real_embed = real_embed.contiguous().view(real_input.size(0), -1, real_embed.size(-1))
            fake_embed = fake_embed.contiguous().view(fake_input.size(0), -1, fake_embed.size(-1))
        else:
            real_embed = real_embed.view(-1, real_input.size(1), real_embed.size(-1))
            fake_embed = fake_embed.view(-1, fake_input.size(1), fake_embed.size(-1))
        real_embed = real_embed.contiguous()
        fake_embed = fake_embed.contiguous()

        real_embed = self.fe_dropout(self.fe_relu(real_embed))
        fake_embed = self.fe_dropout(self.fe_relu(fake_embed))

        if (self.pool_fe_kernel):
            real_embed = real_embed.transpose(1, 2).contiguous()
            real_embed = self.pool_fe(real_embed)
            real_embed = real_embed.transpose(1, 2).contiguous()

            fake_embed = fake_embed.transpose(1, 2).contiguous()
            fake_embed = self.pool_fe(fake_embed)
            fake_embed = fake_embed.transpose(1, 2).contiguous()

        if (self.adaptive_pool_tar_squze_mul):
            real_embed = real_embed.transpose(1, 2).contiguous()
            adaptive_pool_tar_len = real_embed.shape[-1] // self.adaptive_pool_tar_squze_mul
            if (self.pool_fe_type == 'max'):
                real_embed = F.adaptive_max_pool1d(real_embed, adaptive_pool_tar_len)
            elif (self.pool_fe_type == 'avg'):
                real_embed = F.adaptive_avg_pool1d(real_embed, adaptive_pool_tar_len)
            real_embed = real_embed.transpose(1, 2).contiguous()

            fake_embed = fake_embed.transpose(1, 2).contiguous()
            adaptive_pool_tar_len = fake_embed.shape[-1] // self.adaptive_pool_tar_squze_mul
            if (self.pool_fe_type == 'max'):
                fake_embed = F.adaptive_max_pool1d(fake_embed, adaptive_pool_tar_len)
            elif (self.pool_fe_type == 'avg'):
                fake_embed = F.adaptive_avg_pool1d(fake_embed, adaptive_pool_tar_len)
            fake_embed = fake_embed.transpose(1, 2).contiguous()

        self.lstm.flatten_parameters()
        real_r_output, (h_n, h_c) = self.lstm(real_embed)
        fake_r_output, (h_n, h_c) = self.lstm(fake_embed)

        if (self.is_attention):
            real_input_mask = real_input_mask[:, :real_r_output.size(1)]
            fake_input_mask = fake_input_mask[:, :real_r_output.size(1)]
            fake_r_output = fake_r_output[:, :real_r_output.size(1)]

            # transpose batch and sequence (B x S x ..) --> (S x B x ..)
            real_r_output = real_r_output.transpose(0, 1).contiguous()
            fake_r_output = fake_r_output.transpose(0, 1).contiguous()

            attn_output, self.self_attn_weight = self.self_attn(query=real_r_output,
                                                                key=fake_r_output,
                                                                value=fake_r_output,
                                                                key_padding_mask=real_input_mask)
            # transpose batch and sequence (S x B x ..) --> (B x S x ..)
            attn_output = attn_output.transpose(0, 1).contiguous()
            attn_output = torch.sum(attn_output, dim=1).squeeze(dim=1)
            attn_output = F.relu(attn_output)
            attn_output = self.module_fe_dropout(attn_output)

        else:
            attn_output = fake_r_output[:, -1, :]

        return attn_output, self.self_attn_weight

    def fake_embed_self_guide(self, fake_input,
                              fake_input_mask):
        fake_x = fake_input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()

        fake_embed = self.feature_extractor(fake_x).contiguous()
        if (self.batch_first):
            fake_embed = fake_embed.contiguous().view(fake_input.size(0), -1, fake_embed.size(-1))
        else:
            fake_embed = fake_embed.view(-1, fake_input.size(1), fake_embed.size(-1))
        fake_embed = fake_embed.contiguous()

        fake_embed = self.fe_dropout(self.fe_relu(fake_embed))

        if (self.pool_fe_kernel):
            fake_embed = fake_embed.transpose(1, 2).contiguous()
            fake_embed = self.pool_fe(fake_embed)
            fake_embed = fake_embed.transpose(1, 2).contiguous()

        if (self.adaptive_pool_tar_squze_mul):
            fake_embed = fake_embed.transpose(1, 2).contiguous()
            adaptive_pool_tar_len = fake_embed.shape[-1] // self.adaptive_pool_tar_squze_mul
            if (self.pool_fe_type == 'max'):
                fake_embed = F.adaptive_max_pool1d(fake_embed, adaptive_pool_tar_len)
            elif (self.pool_fe_type == 'avg'):
                fake_embed = F.adaptive_avg_pool1d(fake_embed, adaptive_pool_tar_len)
            fake_embed = fake_embed.transpose(1, 2).contiguous()

        self.lstm.flatten_parameters()
        fake_r_output, (h_n, h_c) = self.lstm(fake_embed)

        if (self.is_attention):
            fake_input_mask = fake_input_mask[:, :fake_r_output.size(1)]

            # transpose batch and sequence (B x S x ..) --> (S x B x ..)
            fake_r_output = fake_r_output.transpose(0, 1).contiguous()

            attn_output, self.self_attn_weight = self.self_attn(query=fake_r_output,
                                                                key=fake_r_output,
                                                                value=fake_r_output,
                                                                key_padding_mask=fake_input_mask)
            # transpose batch and sequence (S x B x ..) --> (B x S x ..)
            attn_output = attn_output.transpose(0, 1).contiguous()
            attn_output = torch.sum(attn_output, dim=1).squeeze(dim=1)
            attn_output = F.relu(attn_output)
            attn_output = self.module_fe_dropout(attn_output)

        else:
            attn_output = fake_r_output[:, -1, :]

        return attn_output, self.self_attn_weight
