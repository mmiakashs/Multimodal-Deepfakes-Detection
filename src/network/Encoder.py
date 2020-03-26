from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, original_model, embed_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size

        self.feature_extractor = original_model
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, self.embed_size)
        
        self.dropout = nn.Dropout(0.2)

        self.reset_parameters()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.dropout(F.relu(x))
        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_extractor.fc.weight)
        nn.init.constant_(self.feature_extractor.fc.bias, 0.)