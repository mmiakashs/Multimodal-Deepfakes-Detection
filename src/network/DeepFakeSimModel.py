import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.network.Encoder import Encoder

class DeepFakeSimModel(nn.Module):
    def __init__(self,
                 modalities,
                 modality_embedding_size,
                 module_network='resnet18',
                 fine_tune=True):
        super(DeepFakeSimModel, self).__init__()

        self.modalities = modalities
        self.modality_embedding_size = modality_embedding_size
        self.module_network = module_network
        self.fine_tune = fine_tune
        if(self.module_network=='resnet18'):
            self.original_model = models.resnet18(pretrained=True)
        if (self.fine_tune):
            self.set_parameter_requires_grad(self.original_model, self.fine_tune)

        self.encoder = Encoder(original_model=self.original_model,
                          embed_size=self.modality_embedding_size)
        self.fc = nn.Linear(self.modality_embedding_size,2)

    def set_parameter_requires_grad(self, model, fine_tune):
        for param in model.parameters():
            param.requires_grad = self.fine_tune

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input):
        modality_embed = {}
        modality_output = {}
        for modality in self.modalities:
            shape = input[modality].shape
            tm_input = input[modality].view(-1, shape[-3], shape[-2], shape[-1])
            embed = self.encoder(tm_input)
            embed = embed.contiguous().view(shape[0], -1, embed.size(-1)).contiguous()

            modality_embed[modality] = embed
            output = F.log_softmax(self.fc(embed), dim=1)
            modality_output[modality] = output.mean(dim=1).squeeze(dim=1)

        return modality_output, modality_embed
